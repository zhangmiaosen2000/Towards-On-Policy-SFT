from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import hinted_generate
from utils import NucleusProcessor
from phi_utils import PhiCalculator
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import json, os

SHADOW_SYSTEM_PROMPT_QWEN2 = """
You will be provided with a question and a corresponding ground truth answer which is ensured to be correct.
Your job is to transform the answer to a detailed chain-of-thought (CoT) reasoning process that logically leads to the given answer.
Make sure the content in the CoT is closely matching with the ground truth answer.

Your output should contain a `# Analyze` part to analyze the given solution **line by line**, and then generate a `# CoT` part to provide a complete CoT.
You **MUST** analyze and check every details of the solution in the `# Anayze` first, to make sure fully understand of the problem.
You **MUST NOT** mention the provided answer in the CoT part, as the CoT of original problem does not know the ground truth.
Most importantly, the CoT should follow exactly the language style of your own thinking.
Here is an example:


Input:

# Question
What is the result of sum 1 to 10? please analyze step by step and put the result in \\boxed{}

# Answer:
we can use the formula Sn = (1+n)*n/2.
let n=10, we have:
Sn = (1+10)*10/2 = \\boxed{55}.


Output:

# Analyze

We need to find out how to come up with the idea in the provided answer, lets analyze the provided answer line by line first.

Let's check the first line of the solution, it used a fomula that the results of sum from 1 to N is Sn = (1+N) * N / 2.
This is well known as the Gauss formula and I think we can use it directly.
So when people see this problem, they should start with this formula.

Then the second and third line of the solution put n=10 into the formula and solve the problem.

Now I should transform the solution to my preferred format.

# CoT

To find the sum of the numbers from 1 to 10, we can use the formula for the sum of the first \( n \) natural numbers, which is given by:

\[
S = \\frac{n(n + 1)}{2}
\]

Here, \( n = 10 \). Let's substitute \( n \) into the formula and calculate step by step.

1. Substitute \( n = 10 \) into the formula:
   \[
   S = \frac{10(10 + 1)}{2}
   \]

2. Simplify the expression inside the parentheses:
   \[
   S = \\frac{10 \\times 11}{2}
   \]

3. Perform the multiplication:
   \[
   S = \\frac{110}{2}
   \]

4. Perform the division:
   \[
   S = 55
   \]

So, the sum of the numbers from 1 to 10 is \( 55 \).

\[
\\boxed{55}
\]
"""


INPUTPROMPT = """
# Question:
{question}

# Answer:
{answer}

# CoT:
"""


def gen_fim_cot(model_name, tokenizer, systems, users, answers, args):
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=32
    )

    prompt_list = []
    for system, user, answer in zip(systems, users, answers):
        messages = [{"role": "system", "content": SHADOW_SYSTEM_PROMPT_QWEN2}]
        inp = user if system is None else system + "\n\n" + user
        messages += [
            {"role": "user", "content": INPUTPROMPT.format(question=inp, answer=answer)}
        ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        prompt_list += [{"prompt": text}]

    outputs = llm.generate(prompt_list, sampling_params=SamplingParams(temperature=0, max_tokens=16384))

    cots = [output.outputs[0].text for output in outputs]

    r = []
    for cot in cots:
        ed = cot.rfind("# CoT")
        finished = False
        if ed != -1:
            finished = True
        r.append((cot[:ed + len("# CoT")], finished))

    del llm
    torch.cuda.empty_cache()
    return r

from copy import deepcopy
def HD(tokenizer, jsl, args):
    drafter_model_name = "Qwen/Qwen2.5-7B-Instruct"

    drafter = AutoModelForCausalLM.from_pretrained(drafter_model_name,
            torch_dtype="auto",
            device_map="cuda:0")
    target = AutoModelForCausalLM.from_pretrained(drafter_model_name,
            torch_dtype="auto",
            device_map="cuda:1")


    jsls = [deepcopy(jsl) for _ in range(2)]
    for idx, beta in enumerate([0.0, args.beta]):
        for js in tqdm(jsls[idx]):
            problem = js["problem"]
            answer = js["solution"]
            if "fim_cot" in js:
                fim_cot = js["fim_cot"]
            else:
                continue
            messages = [
                    {"role": "system", "content": SHADOW_SYSTEM_PROMPT_QWEN2},
                    {"role": "user", "content": INPUTPROMPT.format(question=problem, answer=answer)},
                ]
            text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    ) + fim_cot

            input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids_target = input_ids[0].tolist()

            messages = [
                    {"role": "user", "content": problem},
                ]
            text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids_drafter = input_ids[0].tolist()

            gen_len = 8192
            logits_processor = NucleusProcessor(temperature=.3, top_p=.95)

            output_ids_sd, alpha, end_reason = hinted_generate(
                            input_ids_drafter,
                            input_ids_target,
                            drafter,
                            target,
                            tokenizer=tokenizer,
                            spliter="\\boxed{",
                            beta=beta,
                            lambda_mode="linear",
                            sigmoid_center=0.5,
                            piecewise_bounds=(0.3, 0.7),
                            logits_processor=logits_processor,
                            max_gen_len=gen_len,
                            eos_tokens_id_drafter=target.config.eos_token_id,
                            eos_tokens_id_target=target.config.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )

            output_sd = tokenizer.decode(output_ids_sd, skip_special_tokens=True)

            js["cot_tokens_num"] = len(output_ids_sd)
            js["cot"] = output_sd
            js["acceptance_rate"] = alpha
            js["end_reason"] = end_reason
            phi_calculator = PhiCalculator(model=target, tokenizer=tokenizer)

            phi = phi_calculator.compute_phi(
                problem=problem,
                solution=output_sd
            )
            
            extream_phi_r = sum([p > -1 for p in phi.phi_values]) / phi.num_tokens if phi.num_tokens > 0 else 0
            js["extream_phi_-1"] = extream_phi_r
            extream_phi_r = sum([p > -3 for p in phi.phi_values]) / phi.num_tokens if phi.num_tokens > 0 else 0
            js["extream_phi_-3"] = extream_phi_r
            extream_phi_r = sum([p > -5 for p in phi.phi_values]) / phi.num_tokens if phi.num_tokens > 0 else 0
            js["extream_phi_-5"] = extream_phi_r
            js["avg_phi"] = phi.avg_phi

    del target
    del drafter

    return jsls


def original_generate(tokenizer, model_name, problem):
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=32
    )
    messages = [{"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    prompt_list = [{"prompt": text}]
    outputs = llm.generate(prompt_list, sampling_params=SamplingParams(temperature=0, max_tokens=16384))
    output = outputs[0].outputs[0].text
    del llm
    torch.cuda.empty_cache()
    return output





def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()
    return args


DATA={
        "problem": "In a volleyball tournament for the Euro-African cup, there were nine more teams from Europe than from Africa. Each pair of teams played exactly once and the Europeans teams won precisely nine times as many matches as the African teams, overall. What is the maximum number of matches that a single African team might have won?",
        "solution": "1. Let \\( n \\) be the number of African teams. Then, the number of European teams is \\( n + 9 \\).\n2. Each pair of teams plays exactly once. Therefore, the total number of matches played is:\n   \\[\n   \\binom{n}{2} + \\binom{n+9}{2} + n(n+9)\n   \\]\n   where \\(\\binom{n}{2}\\) represents the matches between African teams, \\(\\binom{n+9}{2}\\) represents the matches between European teams, and \\(n(n+9)\\) represents the matches between African and European teams.\n3. Let \\( k \\) be the number of games in which a European team defeated an African team. The condition states that the Europeans won precisely nine times as many matches as the African teams. Therefore, we have:\n   \\[\n   \\binom{n+9}{2} + k = 9 \\left( \\binom{n}{2} + (n(n+9) - k) \\right)\n   \\]\n4. Expanding the binomial coefficients, we get:\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = 9 \\left( \\frac{n(n-1)}{2} + n(n+9) - k \\right)\n   \\]\n5. Simplifying the equation:\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = 9 \\left( \\frac{n(n-1)}{2} + n(n+9) - k \\right)\n   \\]\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = 9 \\left( \\frac{n^2 - n}{2} + n^2 + 9n - k \\right)\n   \\]\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = 9 \\left( \\frac{n^2 - n + 2n^2 + 18n - 2k}{2} \\right)\n   \\]\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = 9 \\left( \\frac{3n^2 + 17n - 2k}{2} \\right)\n   \\]\n   \\[\n   \\frac{(n+9)(n+8)}{2} + k = \\frac{9(3n^2 + 17n - 2k)}{2}\n   \\]\n   \\[\n   (n+9)(n+8) + 2k = 9(3n^2 + 17n - 2k)\n   \\]\n   \\[\n   n^2 + 17n + 72 + 2k = 27n^2 + 153n - 18k\n   \\]\n   \\[\n   2k + 18k = 27n^2 - n^2 + 153n - 17n - 72\n   \\]\n   \\[\n   20k = 26n^2 + 136n - 72\n   \\]\n   \\[\n   k = \\frac{13n^2 + 68n - 36}{10}\n   \\]\n6. Clearly, \\( k \\leq n(n+9) \\), so:\n   \\[\n   10k + 36 \\leq 10n^2 + 90n + 36\n   \\]\n   \\[\n   13n^2 + 68n \\leq 10n^2 + 90n + 36\n   \\]\n   \\[\n   3n^2 - 22n \\leq 36\n   \\]\n   \\[\n   3n^2 - 22n - 36 \\leq 0\n   \\]\n7. Solving the quadratic inequality:\n   \\[\n   n = \\frac{22 \\pm \\sqrt{484 + 432}}{6}\n   \\]\n   \\[\n   n = \\frac{22 \\pm \\sqrt{916}}{6}\n   \\]\n   \\[\n   n = \\frac{22 \\pm 2\\sqrt{229}}{6}\n   \\]\n   \\[\n   n = \\frac{11 \\pm \\sqrt{229}}{3}\n   \\]\n   Since \\( n \\) must be an integer, we test values from 1 to 8. Only \\( n = 6 \\) and \\( n = 8 \\) give integer \\( k \\), which are \\( k = 84 \\) and \\( k = 134 \\) respectively.\n8. If \\( n = 6 \\) and \\( k = 84 \\), there are 6 African teams and 15 European teams. To maximize the number of wins of one African team, have all 6 of these wins be attributed to the same African team. Since they can have up to 5 more wins (from other African teams), the maximum possible is 11.\n9. In the other case of \\( n = 8 \\) and \\( k = 134 \\), the maximum is \\( 7 + 2 = 9 \\), which is not as high.\n\nThus, the answer is 11, achieved when:\n- There are 15 European teams and 6 African teams.\n- One of the African teams wins against all 5 other African teams.\n- That same African team beats 6 out of the 15 European teams. All other games between African and European teams result in a win for the European team.\n\nThis clearly works since:\n\\[\n\\binom{15}{2} + 84 = 189 = 9 \\left( \\binom{6}{2} + 6 \\right)\n\\]\n\nThe final answer is \\(\\boxed{11}\\).",
        "answer": "11",
        "pred": "24"
    }





if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    jsl = [DATA]
    print(">>> Problem:\n", jsl[0]["problem"], "\n\n")
    input(">>> Enter to see the answer...")
    print(">>> Answer:\n", jsl[0]["solution"], "\n\n")
    input(">>> Enter to see the solution of Qwen2.5-7B-Instruct...")

    pred = original_generate(tokenizer, "Qwen/Qwen2.5-7B-Instruct", jsl[0]["problem"])
    print(">>> Qwen2.5-7B-Instruct's solution:\n", pred, "\n\n")

    input(">>> Enter to do Hinted decoding...")

    systems = [None for _ in jsl]
    problems = [js["problem"] for js in jsl]
    answers = [js["solution"] for js in jsl]
    res = gen_fim_cot("Qwen/Qwen2.5-7B-Instruct", tokenizer, systems, problems, answers, args)
    for idx, (fimcot, finished) in enumerate(res):
        if not finished:
            jsl[idx]["fim_ooc"] = True
        else:
            jsl[idx]["fim_cot"] = fimcot
    jsl_beta0, jsl_betak = HD(tokenizer, jsl, args)

    print(">>> Self-distillation baseline (or beta=0) CoT:\n", jsl_beta0[0]["cot"], "\n-  End reason: ", jsl_beta0[0]["end_reason"], "\n-  Distribution info: ", [(k, v) for k, v in jsl_beta0[0].items() if "phi" in k], "\n\n")
    input(">>> Enter to see Hinted decoding result...")
    print(">>> Hinted decoding (beta=", args.beta, ") CoT:\n", jsl_betak[0]["cot"], "\n-  End reason: ", jsl_betak[0]["end_reason"], "\n-  Distribution info: ", [(k, v) for k, v in jsl_betak[0].items() if "phi" in k], "\n\n")