import torch
import torch.nn.functional as F
from torch.nn import Module
from typing import List, Tuple
from collections import defaultdict

from utils.logits_processor import LogitsProcessor, GreedyProcessor
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _ensure_tensor(token_ids, device):
    ids = token_ids if isinstance(token_ids, list) else [token_ids]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)


def entropy_of_probs(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a probability vector, shape [..., V] -> [...]"""
    log_p = torch.log(p + 1e-20)
    return -(p * log_p).sum(dim=-1)


def lambda_linear(H_norm: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.clamp(beta * H_norm, 0.0, 1.0)


def lambda_sigmoid(H_norm: torch.Tensor, beta: float, center: float) -> torch.Tensor:
    return torch.sigmoid(beta * (H_norm - center))


def lambda_piecewise(H_norm: torch.Tensor,
                     h1: float,
                     h2: float) -> torch.Tensor:
    """
    0        if H_norm <= h1
    1        if H_norm >= h2
    linear   else
    """
    zero = torch.zeros_like(H_norm)
    one = torch.ones_like(H_norm)
    lambda_val = (H_norm - h1) / (h2 - h1)
    lambda_val = torch.clamp(lambda_val, 0.0, 1.0)
    return torch.where(H_norm <= h1, zero,
                       torch.where(H_norm >= h2, one, lambda_val))


def has_k_words_repeated_at_least_m(tokens: list, k: int, m: int) -> bool:
    """
    Check if there exists a k-length token subsequence that appears >= m times.
    """
    words = tokens
    n = len(words)
    counts = defaultdict(int)
    for i in range(0, n - k):
        window = tuple(words[i:i + k])
        counts[window] += 1
        if counts[window] >= m:
            return True
    return False


# ---------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------
@torch.no_grad()
def hinted_generate(
    inputs_drafter: List[int],
    inputs_target: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    spliter: str | None = None,
    beta: float = 0.7,
    lambda_mode: str = "linear",
    sigmoid_center: float = 0.5,
    piecewise_bounds: Tuple[float, float] = (0.3, 0.7),
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id_drafter: int | List[int] = 1,
    eos_tokens_id_target: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    first_target: bool = True,
) -> Tuple[List[int], float, str]:
    """
    Decoding with direct geometric mixing  m(x) ~ p(x)^{1-lambda} q(x)^lambda.

    Returns:
        generated_tokens: List[int]
        dummy_rate:      float  (always 1.0, kept for interface compatibility)
        finish_reason:   str    ("EOS-Target"/"EOS-drafter"/"Max-length"/"Repeating")
    """

    # ---------- constants ----------
    drafter_eos = _ensure_tensor(eos_tokens_id_drafter, drafter.device)
    target_eos = _ensure_tensor(eos_tokens_id_target, target.device)

    vocab_size = target.config.vocab_size
    log_vocab = torch.log(torch.tensor(vocab_size, device=target.device))

    drafter_ctx = getattr(drafter.config, "max_position_embeddings",
                          getattr(drafter.config, "max_context_length", 1024))
    target_ctx = getattr(target.config, "max_position_embeddings",
                         getattr(target.config, "max_context_length", 1024))

    # ---------- allocate buffers ----------
    drafter_pref_len = len(inputs_drafter)
    target_pref_len = len(inputs_target)

    drafter_total_len = min(drafter_ctx, drafter_pref_len + max_gen_len)
    target_total_len = min(target_ctx, target_pref_len + max_gen_len)

    drafter_ids = torch.full((1, drafter_total_len), pad_token_id,
                             dtype=torch.long, device=drafter.device)
    target_ids = torch.full((1, target_total_len), pad_token_id,
                             dtype=torch.long, device=target.device)

    drafter_ids[0, :drafter_pref_len] = torch.tensor(
        inputs_drafter, dtype=torch.long, device=drafter.device
    )
    target_ids[0, :target_pref_len] = torch.tensor(
        inputs_target, dtype=torch.long, device=target.device
    )

    drafter_pos = drafter_pref_len
    target_pos = target_pref_len
    generated_len = 0

    drafter_cache: DynamicCache | None = None
    target_cache: DynamicCache | None = None

    generated_text = ""
    drafter_only = False

    # ---------- maybe generate first token purely by target ----------
    if first_target:
        Mp = target(
            input_ids=target_ids[..., :target_pos],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_probs = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_probs)

        target_ids[0, target_pos] = t.to(target.device)
        drafter_ids[0, drafter_pos] = t.to(drafter.device)

        target_pos += 1
        drafter_pos += 1
        generated_len += 1

        if tokenizer is not None:
            generated_text += tokenizer.decode([int(t)])
        if spliter and tokenizer and (spliter in generated_text):
            drafter_only = True

    # ---------- main loop ----------
    printed_len = 0
    while (
        generated_len < max_gen_len
        and drafter_pos < drafter_total_len - 1
        and target_pos < target_total_len - 1
    ):
        if generated_len // 100 > printed_len:
            printed_len = generated_len // 100
            if has_k_words_repeated_at_least_m(target_ids[0, target_pref_len:target_pos].tolist(), k=30, m=5):
                return (
                    target_ids[0, target_pref_len:target_pos].tolist(),
                    1.0,
                    "Repeating"
                )

        if (not drafter_only) and spliter and tokenizer and (spliter in generated_text):
            drafter_only = True

        if drafter_only:
            Md = drafter(
                input_ids=drafter_ids[..., :drafter_pos].to(drafter.device),
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Md.past_key_values
            logits_q = Md.logits[..., -1, :].to(target.device)
            log_q = F.log_softmax(logits_q, dim=-1)
            q_probs = logits_processor(log_q)
            x = logits_processor.sample(q_probs)

            target_ids[0, target_pos] = x.to(target.device)
            drafter_ids[0, drafter_pos] = x.to(drafter.device)
            target_pos += 1
            drafter_pos += 1
            generated_len += 1

            if tokenizer is not None:
                generated_text += tokenizer.decode([int(x)])

            if torch.isin(x, drafter_eos.to(x)):
                return (target_ids[0, target_pref_len:target_pos].tolist(),
                    1.0, "EOS-drafter")
        else:
            # ---- forward target ----
            Mp = target(
                input_ids=target_ids[..., :target_pos],
                past_key_values=target_cache,
                use_cache=use_cache,
            )
            target_cache = Mp.past_key_values
            logits_p = Mp.logits[..., -1, :]

            # ---- forward drafter ----
            Md = drafter(
                input_ids=drafter_ids[..., :drafter_pos].to(drafter.device),
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Md.past_key_values
            logits_q = Md.logits[..., -1, :].to(target.device)

            # ---- lambda(H) ----
            log_p = F.log_softmax(logits_p, dim=-1)
            p_probs = torch.exp(log_p)
            H = entropy_of_probs(p_probs)
            H_norm = H / log_vocab

            if lambda_mode == "linear":
                lam = lambda_linear(H_norm, beta)
            elif lambda_mode == "sigmoid":
                lam = lambda_sigmoid(H_norm, beta, sigmoid_center)
            elif lambda_mode == "piecewise":
                h1, h2 = piecewise_bounds
                lam = lambda_piecewise(H_norm, h1, h2)
            else:
                raise ValueError(f"lambda_mode must be linear/sigmoid/piecewise, got {lambda_mode}")

            # ---- mix & sample ----
            log_q = F.log_softmax(logits_q, dim=-1)

            logits_mix = (1.0 - lam) * log_p + lam * log_q
            mix_probs = logits_processor(logits_mix)
            x = logits_processor.sample(mix_probs)

            # ---- push to sequences ----
            target_ids[0, target_pos] = x.to(target.device)
            drafter_ids[0, drafter_pos] = x.to(drafter.device)

            target_pos += 1
            drafter_pos += 1
            generated_len += 1

            if tokenizer is not None:
                generated_text += tokenizer.decode([int(x)])

            if torch.isin(x, drafter_eos.to(x)):
                return (target_ids[0, target_pref_len:target_pos].tolist(),
                        1.0, "EOS-drafter")

    # ---------- normal finish ----------
    return (
        target_ids[0, target_pref_len:target_pos].tolist(),
        1.0,
        "Max-length"
    )
