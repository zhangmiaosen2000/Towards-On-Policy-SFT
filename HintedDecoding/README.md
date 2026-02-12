# Hinted decoding

## Requirements

```
transformers
vllm
rich
tqdm
termcolor
bitsandbytes==0.43.1
```

## Demo

You can run `python Run_HD_example.py --beta 3` to see an example of a math problem from Numina-Math dataset.
It will show the response of:

- Ground truth solution from the dataset (correct answer = 11)
- The response of Qwen2.5-7B-instruct (which will give an error answer != 11)
- The response of self-distillation (Correct answer with avg-phi <-0.1, which means out-of-distribution)
- The response of hinted decoding (Correct answer with avg-phi ~-0.05, which is much better)

You can try with higher beta.