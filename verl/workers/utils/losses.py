# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def dft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]
    logits = model_output.get("logits")

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        if logits is not None:
            logits_flatten = logits.values()
            probs = torch.softmax(logits_flatten, dim=-1)
            token_probs = torch.exp(log_prob_flatten).detach()
            per_token_loss = -log_prob_flatten * token_probs * loss_mask_flatten
        else:
            token_probs = torch.exp(log_prob_flatten).detach()
            per_token_loss = -log_prob_flatten * token_probs * loss_mask_flatten

        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            token_probs = torch.gather(probs, dim=-1, index=data["responses"].unsqueeze(-1)).squeeze(-1).detach()
        else:
            token_probs = torch.exp(log_prob).detach()
        
        per_token_loss = -log_prob * token_probs * response_mask
        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size

    return loss, {}


def eaft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]
    logits = model_output.get("logits")

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        if logits is not None:
            logits_flatten = logits.values()
            probs = torch.softmax(logits_flatten, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            vocab_size = logits_flatten.size(-1)
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=entropy.dtype, device=entropy.device))
            normalized_entropy = (entropy / max_entropy).detach()
            per_token_loss = -log_prob_flatten * normalized_entropy * loss_mask_flatten
        else:
            per_token_loss = -log_prob_flatten * loss_mask_flatten

        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            vocab_size = logits.size(-1)
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=entropy.dtype, device=entropy.device))
            normalized_entropy = (entropy / max_entropy).detach()
            per_token_loss = -log_prob * normalized_entropy * response_mask
        else:
            per_token_loss = -log_prob * response_mask
        
        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size

    return loss, {}


def indomain_cot_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]
    logits = model_output.get("logits")
    
    tau = getattr(config, 'indomain_tau', 0.0)
    temperature = getattr(config, 'indomain_temperature', 1.0)
    w_min = getattr(config, 'indomain_w_min', 0.1)
    w_max = getattr(config, 'indomain_w_max', 1.0)

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        if logits is not None:
            logits_flatten = logits.values()
            probs = torch.softmax(logits_flatten, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob_flatten + entropy
            
            sigmoid_input = -(phi_t - tau) / temperature
            sigmoid_val = torch.sigmoid(sigmoid_input)
            weight = w_min + (w_max - w_min) * sigmoid_val
            
            per_token_loss = -log_prob_flatten * weight.detach() * loss_mask_flatten
        else:
            per_token_loss = -log_prob_flatten * loss_mask_flatten

        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size
        
        if logits is not None:
            stats = {
                'phi_mean': phi_t[loss_mask_flatten.bool()].mean().item(),
                'phi_std': phi_t[loss_mask_flatten.bool()].std().item(),
                'weight_mean': weight[loss_mask_flatten.bool()].mean().item(),
                'weight_min': weight[loss_mask_flatten.bool()].min().item(),
                'weight_max': weight[loss_mask_flatten.bool()].max().item(),
            }
        else:
            stats = {}
            
    else:
        response_mask = data["response_mask"].to(bool)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob + entropy
            
            sigmoid_input = -(phi_t - tau) / temperature
            sigmoid_val = torch.sigmoid(sigmoid_input)
            weight = w_min + (w_max - w_min) * sigmoid_val
            
            per_token_loss = -log_prob * weight.detach() * response_mask
            
            stats = {
                'phi_mean': phi_t[response_mask].mean().item(),
                'phi_std': phi_t[response_mask].std().item(),
                'weight_mean': weight[response_mask].mean().item(),
                'weight_min': weight[response_mask].min().item(),
                'weight_max': weight[response_mask].max().item(),
            }
        else:
            per_token_loss = -log_prob * response_mask
            stats = {}
        
        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size

    return loss, stats


def adaptive_polylog_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]
    logits = model_output.get("logits")
    
    beta = getattr(config, 'polylog_beta', 1.0)
    gamma_min = getattr(config, 'polylog_gamma_min', 0.1)
    gamma_max = getattr(config, 'polylog_gamma_max', 2.0)

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        if logits is not None:
            logits_flatten = logits.values()
            probs = torch.softmax(logits_flatten, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob_flatten + entropy
            
            gamma_raw = torch.exp(-phi_t / beta)
            gamma_t = torch.clamp(gamma_raw, gamma_min, gamma_max)
            
            token_probs = torch.exp(log_prob_flatten)
            epsilon = 1e-10
            token_probs_safe = torch.clamp(token_probs, epsilon, 1.0)
            
            polylog_loss_val = -(token_probs_safe ** gamma_t) * torch.log(token_probs_safe)
            
            per_token_loss = polylog_loss_val.detach() * loss_mask_flatten
        else:
            per_token_loss = -log_prob_flatten * loss_mask_flatten

        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size
        
        if logits is not None:
            valid_mask = loss_mask_flatten.bool()
            stats = {
                'phi_mean': phi_t[valid_mask].mean().item(),
                'phi_std': phi_t[valid_mask].std().item(),
                'phi_min': phi_t[valid_mask].min().item(),
                'phi_max': phi_t[valid_mask].max().item(),
                'gamma_mean': gamma_t[valid_mask].mean().item(),
                'gamma_std': gamma_t[valid_mask].std().item(),
                'gamma_min': gamma_t[valid_mask].min().item(),
                'gamma_max': gamma_t[valid_mask].max().item(),
            }
        else:
            stats = {}
            
    else:
        response_mask = data["response_mask"].to(bool)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob + entropy
            
            gamma_raw = torch.exp(-phi_t / beta)
            gamma_t = torch.clamp(gamma_raw, gamma_min, gamma_max)
            
            token_probs = torch.exp(log_prob)
            epsilon = 1e-10
            token_probs_safe = torch.clamp(token_probs, epsilon, 1.0)
            
            polylog_loss_val = -(token_probs_safe ** gamma_t) * torch.log(token_probs_safe)
            
            per_token_loss = polylog_loss_val.detach() * response_mask
            
            stats = {
                'phi_mean': phi_t[response_mask].mean().item(),
                'phi_std': phi_t[response_mask].std().item(),
                'phi_min': phi_t[response_mask].min().item(),
                'phi_max': phi_t[response_mask].max().item(),
                'gamma_mean': gamma_t[response_mask].mean().item(),
                'gamma_std': gamma_t[response_mask].std().item(),
                'gamma_min': gamma_t[response_mask].min().item(),
                'gamma_max': gamma_t[response_mask].max().item(),
            }
        else:
            per_token_loss = -log_prob * response_mask
            stats = {}
        
        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size

    return loss, stats


def reverse_polylog_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]
    logits = model_output.get("logits")
    
    beta = getattr(config, 'reverse_polylog_beta', 1.0)
    gamma_min = getattr(config, 'reverse_polylog_gamma_min', 0.1)
    gamma_max = getattr(config, 'reverse_polylog_gamma_max', 2.0)

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]
        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        if logits is not None:
            logits_flatten = logits.values()
            probs = torch.softmax(logits_flatten, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob_flatten + entropy
            
            gamma_raw = torch.exp(-phi_t / beta)
            gamma_t = torch.clamp(gamma_raw, gamma_min, gamma_max)
            
            token_probs = torch.exp(log_prob_flatten)
            epsilon = 1e-10
            token_probs_safe = torch.clamp(token_probs, epsilon, 1.0 - epsilon)
            
            reverse_weight = (1.0 - token_probs_safe) ** gamma_t
            reverse_polylog_loss_val = reverse_weight * (-torch.log(token_probs_safe))
            
            per_token_loss = reverse_polylog_loss_val.detach() * loss_mask_flatten
        else:
            per_token_loss = -log_prob_flatten * loss_mask_flatten

        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size
        
        if logits is not None:
            valid_mask = loss_mask_flatten.bool()
            stats = {
                'phi_mean': phi_t[valid_mask].mean().item(),
                'phi_std': phi_t[valid_mask].std().item(),
                'phi_min': phi_t[valid_mask].min().item(),
                'phi_max': phi_t[valid_mask].max().item(),
                'gamma_mean': gamma_t[valid_mask].mean().item(),
                'gamma_std': gamma_t[valid_mask].std().item(),
                'gamma_min': gamma_t[valid_mask].min().item(),
                'gamma_max': gamma_t[valid_mask].max().item(),
                'weight_mean': reverse_weight[valid_mask].mean().item(),
                'weight_std': reverse_weight[valid_mask].std().item(),
            }
        else:
            stats = {}
            
    else:
        response_mask = data["response_mask"].to(bool)
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            
            phi_t = log_prob + entropy
            
            gamma_raw = torch.exp(-phi_t / beta)
            gamma_t = torch.clamp(gamma_raw, gamma_min, gamma_max)
            
            token_probs = torch.exp(log_prob)
            epsilon = 1e-10
            token_probs_safe = torch.clamp(token_probs, epsilon, 1.0 - epsilon)
            
            reverse_weight = (1.0 - token_probs_safe) ** gamma_t
            reverse_polylog_loss_val = reverse_weight * (-torch.log(token_probs_safe))
            
            per_token_loss = reverse_polylog_loss_val.detach() * response_mask
            
            stats = {
                'phi_mean': phi_t[response_mask].mean().item(),
                'phi_std': phi_t[response_mask].std().item(),
                'phi_min': phi_t[response_mask].min().item(),
                'phi_max': phi_t[response_mask].max().item(),
                'gamma_mean': gamma_t[response_mask].mean().item(),
                'gamma_std': gamma_t[response_mask].std().item(),
                'gamma_min': gamma_t[response_mask].min().item(),
                'gamma_max': gamma_t[response_mask].max().item(),
                'weight_mean': reverse_weight[response_mask].mean().item(),
                'weight_std': reverse_weight[response_mask].std().item(),
            }
        else:
            per_token_loss = -log_prob * response_mask
            stats = {}
        
        loss = torch.sum(per_token_loss) / batch_num_tokens * dp_size

    return loss, stats


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: model output tensor of shape [bsz, 1]
        data: TensorDict with "prompt_ids", "response_ids", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = response_ids.offsets().max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = _slice_response_from_unpad_output(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = _slice_response_from_unpad_output(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
