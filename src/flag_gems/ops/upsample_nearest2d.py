"""Triton implementation of upsample_nearest2d forward and backward operators.

Provides GPU-accelerated nearest-neighbor 2D upsampling/downsampling using Triton.
Supports output_size, scales_h/scales_w parameters, and gradient computation.
Optimized for Iluvatar BI-V150 GPU compatibility.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"),
    key=["N", "C", "OH", "OW"],
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    # Each program handles BLOCK_SIZE output pixels across all N*C channels
    if USE_INT32_IDX:
        pid0 = tl.program_id(axis=0)
    else:
        pid0 = tl.program_id(axis=0).to(tl.int64)

    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    idx = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_spatial_size = OH * OW
    mask = idx < total_spatial_size

    # Decompose linear index to (oh, ow)
    ow = idx % OW
    oh = idx // OW

    # Map output coordinates to input coordinates via nearest-neighbor
    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32),
            IH - 1,
        )

    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32),
            IW - 1,
        )

    # Pre-compute spatial offset for input (per-pixel, shared across channels)
    spatial_offset_i = ih * stride_h + iw * stride_w

    # Output offset (output is always contiguous)
    offset_o = nc_iter * (OH * OW) + idx

    # Input offset using strides (handles non-contiguous input)
    n = nc_iter // C
    c = nc_iter % C
    offset_i = n * stride_n + c * stride_c + spatial_offset_i

    dst_nc_stride = nc_stride * (OH * OW)

    # Iterate over N*C channels with stride for memory coalescing
    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i, mask=mask)
        tl.store(ptr_o + offset_o, data, mask=mask)

        nc_iter += nc_stride
        offset_o += dst_nc_stride
        n = nc_iter // C
        c = nc_iter % C
        offset_i = n * stride_n + c * stride_c + spatial_offset_i


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"),
    key=["N", "C", "IH", "IW"],
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_go,
    ptr_gi,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    # Each program handles BLOCK_SIZE input pixels, accumulating gradients
    # from all output pixels that map to each input pixel via nearest-neighbor
    if USE_INT32_IDX:
        pid0 = tl.program_id(axis=0)
    else:
        pid0 = tl.program_id(axis=0).to(tl.int64)

    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    idx = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_spatial_size = IH * IW
    mask = idx < total_spatial_size

    # Decompose linear index to (ih, iw)
    iw = idx % IW
    ih = idx // IW

    # Compute range of output pixels that map to this input pixel
    # For nearest: oh maps to ih if floor(oh * rsh) == ih
    # So oh_start = ceil(ih * rsh), oh_end = ceil((ih+1) * rsh)
    if SAME_H:
        oh_start = ih
        oh_end = ih + 1
    else:
        oh_start = tl.maximum(
            tl.math.ceil(ih.to(tl.float32) * reciprocal_scale_h).to(tl.int32), 0
        )
        oh_end = tl.minimum(
            tl.math.ceil((ih.to(tl.float32) + 1.0) * reciprocal_scale_h).to(tl.int32),
            OH,
        )

    if SAME_W:
        ow_start = iw
        ow_end = iw + 1
    else:
        ow_start = tl.maximum(
            tl.math.ceil(iw.to(tl.float32) * reciprocal_scale_w).to(tl.int32), 0
        )
        ow_end = tl.minimum(
            tl.math.ceil((iw.to(tl.float32) + 1.0) * reciprocal_scale_w).to(tl.int32),
            OW,
        )

    offset_gi = nc_iter * (IH * IW) + idx
    go_nc_base = nc_iter * (OH * OW)
    src_nc_stride = nc_stride * (OH * OW)
    dst_nc_stride = nc_stride * (IH * IW)

    # Iterate over N*C channels for gradient accumulation
    while nc_iter < NC:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        # Accumulate gradients from all contributing output pixels
        # This loop iterates over the output pixels that map to each input pixel
        for oh in range(oh_start, oh_end):
            for ow in range(ow_start, ow_end):
                go_offset = go_nc_base + oh.to(tl.int64) * OW + ow.to(tl.int64)
                val = tl.load(ptr_go + go_offset, mask=mask, other=0.0)
                acc += val

        tl.store(ptr_gi + offset_gi, acc.to(ptr_go.dtype.element_ty), mask=mask)

        go_nc_base += src_nc_stride
        offset_gi += dst_nc_stride
        nc_iter += nc_stride


def _calculate_scale(in_sz: int, out_sz: int, s: Optional[float]) -> float:
    """Compute reciprocal scale factor matching PyTorch's precision.

    Uses torch.tensor for float division to ensure consistent rounding
    behavior with PyTorch's native implementation.
    """
    if s is not None:
        return float(torch.tensor(1.0 / s, dtype=torch.float32).item())
    return float(
        (
            torch.tensor(in_sz, dtype=torch.float32)
            / torch.tensor(out_sz, dtype=torch.float32)
        ).item()
    )


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """Nearest-neighbor 2D upsampling using Triton.

    Args:
        input: Input tensor of shape (N, C, H, W).
        output_size: Target spatial size (OH, OW).
        scales_h: Scale factor for height dimension.
        scales_w: Scale factor for width dimension.

    Returns:
        Output tensor of shape (N, C, OH, OW).
    """
    logger.debug("GEMS UPSAMPLE NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape

    # Handle empty tensors
    if N == 0 or C == 0 or OH == 0 or OW == 0:
        return torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    reciprocal_scale_h = _calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = _calculate_scale(IW, OW, scales_w)

    # Get input strides (handles non-contiguous tensors natively)
    strides = input.stride()
    stride_n = strides[0]
    stride_c = strides[1]
    stride_h = strides[2]
    stride_w = strides[3]

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = OH * OW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            stride_n,
            stride_c,
            stride_h,
            stride_w,
        )
    return output


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int],
    input_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """Backward pass for nearest-neighbor 2D upsampling.

    Computes gradient with respect to input by accumulating gradients from
    all output pixels that map to each input pixel.

    Args:
        grad_output: Gradient tensor of shape (N, C, OH, OW).
        output_size: Target spatial size (OH, OW).
        input_size: Original input size (N, C, H, W).
        scales_h: Scale factor for height dimension.
        scales_w: Scale factor for width dimension.

    Returns:
        Gradient with respect to input, shape (N, C, H, W).
    """
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    N, C, IH, IW = input_size
    OH, OW = output_size

    # Handle empty tensors
    if N == 0 or C == 0 or IH == 0 or IW == 0:
        return torch.zeros(
            input_size, device=grad_output.device, dtype=grad_output.dtype
        )

    reciprocal_scale_h = _calculate_scale(IH, OH, scales_h)
    reciprocal_scale_w = _calculate_scale(IW, OW, scales_w)

    # Use torch.empty since the kernel writes to all elements
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )
    total_threads = IH * IW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output,
            grad_input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return grad_input
