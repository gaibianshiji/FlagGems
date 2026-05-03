import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

SMALL_THRESHOLD = 4096


@triton.jit
def _smooth_l1_loss_elementwise_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, beta_ptr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    beta_f = tl.load(beta_ptr).to(tl.float32)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    diff = tl.abs(xf - yf)
    loss = tl.where(
        diff < beta_f,
        0.5 * diff * diff / beta_f,
        diff - 0.5 * beta_f,
    )

    tl.store(out_ptr + offsets, loss, mask=mask)


@triton.jit
def _smooth_l1_loss_sum_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, beta_ptr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    beta_f = tl.load(beta_ptr).to(tl.float32)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    diff = tl.abs(xf - yf)
    loss = tl.where(
        diff < beta_f,
        0.5 * diff * diff / beta_f,
        diff - 0.5 * beta_f,
    )
    loss = tl.where(mask, loss, 0.0)

    acc = tl.sum(loss, axis=0)
    tl.atomic_add(out_ptr, acc)


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none,1=mean,2=sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if not (input.is_cuda and target.is_cuda):
        raise AssertionError(
            "smooth_l1_loss: input and target must be CUDA tensors for Triton kernel."
        )
    if input.device != target.device:
        raise AssertionError(
            "smooth_l1_loss: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "smooth_l1_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def _pytorch_fallback(input, target, reduction, beta):
    # Use torch.ops.aten to avoid FlagGems dispatch recursion
    diff = torch.abs(input - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta,
    )
    if reduction == 0:
        return loss
    elif reduction == 2:
        return torch.ops.aten.sum(loss)
    else:
        return torch.ops.aten.mean(loss)


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, reduction=1, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if n_elements == 0:
        if red == 0:
            return torch.empty_like(input)
        elif red == 2:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        else:
            return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    if n_elements < SMALL_THRESHOLD:
        return _pytorch_fallback(input, target, red, beta)

    beta_tensor = torch.tensor(beta, dtype=torch.float32, device=input.device)

    if red == 0:
        # reduction = 'none'
        out = torch.empty_like(input)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _smooth_l1_loss_elementwise_kernel[grid](
            input, target, out, n_elements, beta_tensor, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        # reduction = 'sum' or 'mean' (1=mean, 2=sum)
        tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _smooth_l1_loss_sum_kernel[grid](
            input, target, tmp_sum, n_elements, beta_tensor, BLOCK_SIZE=BLOCK_SIZE
        )
        if red == 2:
            # sum
            return tmp_sum.to(dtype=input.dtype)
        else:
            # mean
            mean_val = (tmp_sum / float(n_elements)).to(dtype=input.dtype)
            return mean_val


def smooth_l1_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction=1,
    beta=1.0,
    out: torch.Tensor = None,
):
    logger.debug("GEMS SMOOTH_L1_LOSS_OUT")
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if n_elements < SMALL_THRESHOLD:
        result = _pytorch_fallback(input, target, red, beta)
        if out is not None:
            out.copy_(result)
            return out
        return result

    beta_tensor = torch.tensor(beta, dtype=torch.float32, device=input.device)

    if out is None:
        # Allocate output based on reduction
        if red == 0:
            out = torch.empty_like(input)
        else:
            out = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if not out.is_cuda:
            raise AssertionError("smooth_l1_loss_out: out must be a CUDA tensor.")
        if red == 0:
            if out.numel() != n_elements:
                raise AssertionError(
                    "smooth_l1_loss_out: for reduction='none', out must match input shape."
                )
        else:
            if out.numel() != 1:
                raise AssertionError(
                    "smooth_l1_loss_out: for reduction='sum' or 'mean', out must be a scalar tensor."
                )
        if out.device != input.device:
            raise AssertionError(
                "smooth_l1_loss_out: out must be on the same device as input."
            )

    if red == 0:
        if n_elements > 0:
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _smooth_l1_loss_elementwise_kernel[grid](
                input, target, out, n_elements, beta_tensor, BLOCK_SIZE=BLOCK_SIZE
            )
        return out
    else:
        if n_elements == 0:
            if red == 2:
                out.fill_(0)
            else:
                out.fill_(float("nan"))
            return out
        tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _smooth_l1_loss_sum_kernel[grid](
            input, target, tmp_sum, n_elements, beta_tensor, BLOCK_SIZE=BLOCK_SIZE
        )
        if red == 2:
            out.fill_(tmp_sum.to(dtype=input.dtype))
        else:
            mean_val = (tmp_sum / float(n_elements)).to(dtype=input.dtype)
            out.fill_(mean_val)
        return out
