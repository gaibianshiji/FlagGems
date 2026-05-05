"""CTC Loss operator using Triton."""

from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_MAX_BLOCK = 4096


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


_REDUCTION_MAP = {0: "none", 1: "mean", 2: "sum"}


@triton.jit
def log_sum_exp_2_vec(a, b):
    """Numerically stable log(exp(a) + exp(b)) for two vectors."""
    max_val = tl.maximum(a, b)
    both_inf = (a == -float("inf")) & (b == -float("inf"))
    result = max_val + tl.log(tl.exp(a - max_val) + tl.exp(b - max_val))
    return tl.where(both_inf, -float("inf"), result)


@triton.jit
def log_sum_exp_3_vec(a, b, c, use_c):
    """Numerically stable log(exp(a) + exp(b) + exp(c)) for three vectors."""
    max_ab = tl.maximum(a, b)
    max_val = tl.where(use_c, tl.maximum(max_ab, c), max_ab)
    sum_exp = tl.exp(a - max_val) + tl.exp(b - max_val)
    sum_exp = tl.where(use_c, sum_exp + tl.exp(c - max_val), sum_exp)
    all_inf = (a == -float("inf")) & (b == -float("inf"))
    all_inf = tl.where(use_c, all_inf & (c == -float("inf")), all_inf)
    result = max_val + tl.log(sum_exp)
    return tl.where(all_inf, -float("inf"), result)


@libentry()
@triton.jit
def ctc_loss_kernel(
    log_probs_ptr,  # (T, B, C) contiguous
    targets_ptr,  # (B, S) contiguous
    input_lengths_ptr,  # (B,) contiguous
    target_lengths_ptr,  # (B,) contiguous
    losses_ptr,  # (B,) output
    workspace_ptr,  # (B, BLOCK) workspace for shifted alpha values
    T: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    blank: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """CTC forward kernel. Each program handles one batch element.

    Optimized: keeps alpha_s in registers across time steps, only reads
    shifted values (alpha[s-1], alpha[s-2]) from workspace. Initializes
    workspace inside kernel to eliminate external reset overhead.
    """
    pid_b = tl.program_id(0)
    NEG_INF = -float("inf")

    input_length = tl.load(input_lengths_ptr + pid_b).to(tl.int32)
    target_length = tl.load(target_lengths_ptr + pid_b).to(tl.int32)
    extended_length = 2 * target_length + 1

    s_offsets = tl.arange(0, BLOCK)
    pos_mask = s_offsets < extended_length

    # Build extended target: [blank, y0, blank, y1, ..., blank, y_{U-1}, blank]
    is_even = (s_offsets % 2) == 0
    target_idx = s_offsets // 2
    target_val = tl.load(targets_ptr + pid_b * S + target_idx, mask=target_idx < target_length, other=0)
    ext_target = tl.where(is_even, blank, target_val)

    # Pre-compute can_skip: s>=2, ext_target[s]!=blank, ext_target[s]!=ext_target[s-2]
    s_m2 = s_offsets - 2
    is_even_m2 = (s_m2 % 2) == 0
    target_idx_m2 = s_m2 // 2
    target_val_m2 = tl.load(
        targets_ptr + pid_b * S + target_idx_m2,
        mask=(target_idx_m2 >= 0) & (target_idx_m2 < target_length),
        other=0,
    )
    ext_target_m2 = tl.where(is_even_m2, blank, target_val_m2)
    can_skip = (s_offsets >= 2) & pos_mask & (ext_target != blank) & (ext_target != ext_target_m2)

    # Initialize workspace to -inf (eliminates external reset)
    ws_base = pid_b * BLOCK
    tl.store(
        workspace_ptr + ws_base + s_offsets,
        tl.full([BLOCK], NEG_INF, dtype=tl.float32),
        mask=pos_mask,
    )

    # Initialize alpha for t=0
    log_prob_blank_0 = tl.load(log_probs_ptr + pid_b * C + blank)
    alpha_s = tl.full([BLOCK], NEG_INF, dtype=tl.float32)
    alpha_s = tl.where(s_offsets == 0, log_prob_blank_0, alpha_s)
    first_target = tl.load(targets_ptr + pid_b * S, mask=target_length > 0, other=0)
    log_prob_tgt_0 = tl.load(log_probs_ptr + pid_b * C + first_target, mask=target_length > 0, other=NEG_INF)
    alpha_s = tl.where((s_offsets == 1) & (target_length > 0), log_prob_tgt_0, alpha_s)
    tl.store(workspace_ptr + ws_base + s_offsets, alpha_s, mask=pos_mask)

    # Time loop: alpha_s stays in registers, only load shifted values from workspace
    for t in tl.range(1, T):
        if t < input_length:
            base_t = t * B * C + pid_b * C
            alpha_s_1 = tl.load(
                workspace_ptr + ws_base + s_offsets - 1,
                mask=(s_offsets >= 1) & pos_mask,
                other=NEG_INF,
            )
            alpha_s_2 = tl.load(
                workspace_ptr + ws_base + s_offsets - 2,
                mask=(s_offsets >= 2) & pos_mask,
                other=NEG_INF,
            )
            log_probs_s = tl.load(log_probs_ptr + base_t + ext_target, mask=pos_mask, other=NEG_INF)
            lse = log_sum_exp_3_vec(alpha_s, alpha_s_1, alpha_s_2, can_skip)
            alpha_s = lse + log_probs_s
            tl.store(workspace_ptr + ws_base + s_offsets, alpha_s, mask=pos_mask)

    # Final loss
    idx_last = extended_length - 1
    idx_prev = extended_length - 2
    alpha_last = tl.load(workspace_ptr + ws_base + idx_last, mask=extended_length >= 1, other=NEG_INF)
    alpha_prev_last = tl.load(workspace_ptr + ws_base + idx_prev, mask=extended_length >= 2, other=NEG_INF)
    loss = -log_sum_exp_2_vec(alpha_last, alpha_prev_last)
    loss = tl.where(extended_length < 2, -alpha_last, loss)
    tl.store(losses_ptr + pid_b, loss)


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction=Reduction.MEAN.value,
    zero_infinity=False,
):
    T, B, C = log_probs.shape
    S = targets.shape[1] if targets.dim() > 1 else targets.shape[0]

    log_probs = log_probs.contiguous()
    targets = targets.contiguous()
    input_lengths = input_lengths.contiguous()
    target_lengths = target_lengths.contiguous()

    losses = torch.empty(B, device=log_probs.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(2 * S + 1)
    assert BLOCK <= _MAX_BLOCK, (
        f"Extended target length {2 * S + 1} exceeds max BLOCK {_MAX_BLOCK}. "
        f"Reduce target sequence length (max S={(_MAX_BLOCK - 1) // 2})."
    )
    workspace = torch.empty(B, BLOCK, device=log_probs.device, dtype=torch.float32)

    with torch_device_fn.device(log_probs.device):
        ctc_loss_kernel[(B,)](
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            losses,
            workspace,
            T=T,
            B=B,
            C=C,
            S=S,
            blank=blank,
            BLOCK=BLOCK,
        )

    if zero_infinity:
        losses = torch.where(torch.isinf(losses) | torch.isnan(losses), torch.zeros_like(losses), losses)

    reduction_str = _REDUCTION_MAP.get(reduction, "mean")
    out_dtype = log_probs.dtype
    if reduction_str == "mean":
        tl_float = target_lengths.float().clamp(min=1.0)
        result = (losses / tl_float).mean().to(out_dtype)
    elif reduction_str == "sum":
        result = losses.sum().to(out_dtype)
    else:
        result = losses.to(out_dtype)
    return result
