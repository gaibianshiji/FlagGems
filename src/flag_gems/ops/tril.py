import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    row_start = pid_m * M_BLOCK_SIZE
    col_start = pid_n * N_BLOCK_SIZE

    row = row_start + tl.arange(0, M_BLOCK_SIZE)[:, None]
    col = col_start + tl.arange(0, N_BLOCK_SIZE)[None, :]
    mask = (row < M) & (col < N)

    x = tl.load(X + row * N + col, mask=mask, other=0.0)
    y = tl.where(col <= row + diagonal, x, 0.0)
    tl.store(Y + row * N + col, y, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("tril_batch"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_batch_kernel(
    X,
    Y,
    B_M,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (B*M_blocks, N_blocks) - flattened batch*row dimension
    pid_bm = tle.program_id(0)
    pid_n = tle.program_id(1)

    batch_id = pid_bm // ((M + M_BLOCK_SIZE - 1) // M_BLOCK_SIZE)
    pid_m = pid_bm - batch_id * ((M + M_BLOCK_SIZE - 1) // M_BLOCK_SIZE)

    row_start = pid_m * M_BLOCK_SIZE
    col_start = pid_n * N_BLOCK_SIZE

    rows = row_start + tl.arange(0, M_BLOCK_SIZE)[:, None]
    cols = col_start + tl.arange(0, N_BLOCK_SIZE)[None, :]

    m_mask = rows < M
    n_mask = cols < N
    mask = m_mask & n_mask

    x_offsets = batch_id * M * N + rows * N + cols
    x = tl.load(X + x_offsets, mask=mask, other=0.0)
    y = tl.where(cols <= rows + diagonal, x, 0.0)
    tl.store(Y + x_offsets, y, mask=mask)


def tril(A, diagonal=0):
    """Lower triangular part of a matrix (or batch of matrices).

    Args:
        A: Input tensor with at least 2 dimensions.
        diagonal: Diagonal offset. 0 = main diagonal, >0 = above, <0 = below.

    Returns:
        Tensor with elements above the specified diagonal set to zero.
    """
    logger.debug("GEMS TRIL")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    if A.is_contiguous():
        A_input = A
    else:
        A_input = A.contiguous()

    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)

    M, N = A_input.shape[-2:]

    if M == 0 or N == 0:
        return out

    with torch_device_fn.device(A_input.device):
        if len(A_input.shape) == 2:
            grid = lambda meta: (
                triton.cdiv(M, meta["M_BLOCK_SIZE"]),
                triton.cdiv(N, meta["N_BLOCK_SIZE"]),
            )
            tril_kernel[grid](A_input, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A_input) / M / N)
            grid = lambda meta: (
                batch * triton.cdiv(M, meta["M_BLOCK_SIZE"]),
                triton.cdiv(N, meta["N_BLOCK_SIZE"]),
            )
            tril_batch_kernel[grid](A_input, out, batch, M, N, diagonal)
            out = out.view(A.shape)

    return out


def tril_(A, diagonal=0):
    """In-place version of tril. Modifies A directly.

    Args:
        A: Input tensor with at least 2 dimensions (modified in-place).
        diagonal: Diagonal offset.

    Returns:
        The modified input tensor A.
    """
    logger.debug("GEMS TRIL_ (inplace)")
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    result = tril(A, diagonal)
    A.copy_(result)
    return A


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    """tril with explicit output tensor.

    Args:
        input: Input tensor with at least 2 dimensions.
        diagonal: Diagonal offset.
        out: Output tensor (optional, created if None).

    Returns:
        Output tensor with lower triangular elements.
    """
    if out is None:
        out = torch.empty_like(input)
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    result = tril(input, diagonal)
    out.copy_(result)
    return out
