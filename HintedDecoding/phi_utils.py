import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PhiResult:
    phi_values: np.ndarray
    log_probs: np.ndarray
    entropies: np.ndarray
    token_probs: np.ndarray
    token_ids: np.ndarray
    tokens: List[str]

    total_phi: float
    avg_phi: float
    num_tokens: int
    positive_ratio: float
    cumsum_phi: np.ndarray

    problem: str
    solution: str

    def to_dict(self) -> Dict:
        return {
            'phi_values': self.phi_values.tolist(),
            'log_probs': self.log_probs.tolist(),
            'entropies': self.entropies.tolist(),
            'token_probs': self.token_probs.tolist(),
            'token_ids': self.token_ids.tolist(),
            'tokens': self.tokens,
            'total_phi': float(self.total_phi),
            'avg_phi': float(self.avg_phi),
            'num_tokens': int(self.num_tokens),
            'positive_ratio': float(self.positive_ratio),
            'cumsum_phi': self.cumsum_phi.tolist(),
            'problem': self.problem,
            'solution': self.solution
        }

    def get_part_info(self, type="qwen"):
        '''
        Get the cot and answer part separately.
        cot part does not contain <think> and </think>.
        Put this function in try except, as data may not complete and contain <think> token.
        '''
        if type == "qwen":
            start_idx = np.where(self.token_ids == 151667)[0][0]  # <think>
            end_idx = np.where(self.token_ids == 151668)[0][0]    # </think>
            dc = self.to_dict()
            cot_part = dict()
            answer_part = dict()
            for k, v in dc.items():
                if isinstance(v, list):
                    cot_part[k] = v[start_idx + 1: end_idx]
                    answer_part[k] = v[end_idx + 1:]
                else:
                    cot_part[k] = v
                    answer_part[k] = v
        else:
            raise NotImplementedError

        for k in ["total_phi", "avg_phi", "num_tokens", "positive_ratio", "cumsum_phi"]:
            cot_part.pop(k)
            answer_part.pop(k)

        for part in [cot_part, answer_part]:
            part["num_tokens"] = len(part["phi_values"])
            part["total_phi"] = sum(part["phi_values"])
            part["avg_phi"] = part["total_phi"] / part["num_tokens"] if part["num_tokens"] > 0 else 0
            part["positive_ratio"] = sum([p > 0 for p in part["phi_values"]]) / part["num_tokens"] if part["num_tokens"] > 0 else 0

        return cot_part, answer_part


class PhiCalculator:
    def __init__(self, model, tokenizer, temperature: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature if temperature > 0 else 1e-8
        self.device = model.device

    def compute_phi(
        self,
        problem: str,
        solution: str,
        return_tokens: bool = True,
        prompt_template: str = "chat",
        enable_thinking: bool = True
    ) -> PhiResult:
        """
        phi(t) = log p_tau(x_t | context) + H_tau(context)
        """
        if prompt_template is None:
            prompt = f"Problem: {problem}\n\nSolution: "
        elif prompt_template == "":
            prompt = problem
        elif prompt_template == "chat":
            messages = [
                {"role": "user", "content": problem},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        else:
            prompt = prompt_template.format(question=problem)

        full_text = prompt + solution

        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        full_encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True
        )

        prompt_length = prompt_encoding.input_ids.shape[1]
        full_input_ids = full_encoding.input_ids.to(self.device)

        answer_length = full_input_ids.shape[1] - prompt_length
        if answer_length <= 0:
            return PhiResult(
                phi_values=np.array([]),
                log_probs=np.array([]),
                entropies=np.array([]),
                token_probs=np.array([]),
                token_ids=np.array([]),
                tokens=[],
                total_phi=0.0,
                avg_phi=0.0,
                num_tokens=0,
                positive_ratio=0.0,
                cumsum_phi=np.array([]),
                problem=problem,
                solution=solution
            )

        with torch.inference_mode():
            outputs = self.model(full_input_ids)
            logits = outputs.logits

        answer_logits = logits[0, prompt_length - 1:-1, :]
        answer_token_ids = full_input_ids[0, prompt_length:]

        if answer_logits.dtype == torch.bfloat16:
            answer_logits = answer_logits.float()

        scaled_logits = answer_logits / self.temperature

        probs = F.softmax(scaled_logits, dim=-1)
        log_probs_all = F.log_softmax(scaled_logits, dim=-1)

        entropies = -torch.sum(probs * log_probs_all, dim=-1)

        token_log_probs = log_probs_all.gather(
            dim=-1,
            index=answer_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        token_probs = probs.gather(
            dim=-1,
            index=answer_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        phi_values = token_log_probs + entropies
        phi_values_np = phi_values.float().cpu().numpy()
        log_probs_np = token_log_probs.float().cpu().numpy()
        entropies_np = entropies.float().cpu().numpy()
        token_probs_np = token_probs.float().cpu().numpy()
        token_ids_np = answer_token_ids.cpu().numpy()

        cumsum_phi = np.cumsum(phi_values_np)

        tokens = []
        if return_tokens:
            tokens = [
                self.tokenizer.decode([tid])
                for tid in token_ids_np
            ]

        total_phi = float(np.sum(phi_values_np))
        avg_phi = float(total_phi / len(phi_values_np)) if len(phi_values_np) > 0 else 0.0
        positive_ratio = float(np.mean(phi_values_np > 0)) if len(phi_values_np) > 0 else 0.0

        return PhiResult(
            phi_values=phi_values_np,
            log_probs=log_probs_np,
            entropies=entropies_np,
            token_probs=token_probs_np,
            token_ids=token_ids_np,
            tokens=tokens,
            total_phi=total_phi,
            avg_phi=avg_phi,
            num_tokens=len(phi_values_np),
            positive_ratio=positive_ratio,
            cumsum_phi=cumsum_phi,
            problem=problem,
            solution=solution
        )
