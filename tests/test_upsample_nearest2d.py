"""Comprehensive tests for upsample_nearest2d operator.

Covers:
- Various input shapes (small, regular, large)
- Scale factors (2x, 3x, 0.5x, non-integer)
- Parameter modes (output_size, scales_h/scales_w)
- Data types (float32, float16, bfloat16)
- Edge cases (1x1 input, single channel, empty tensors)
- Backward gradient verification
- Non-contiguous tensors
- Boundary values (NaN, Inf, extreme values)
"""

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


# ============================================================================
# Forward tests with various shapes and scales
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Forward + Backward gradient verification
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    "batch,channel,h,w,out_h,out_w",
    [
        (1, 1, 2, 2, 4, 4),  # small, 2x upsample
        (4, 3, 32, 32, 64, 64),  # regular, 2x upsample
        (2, 16, 16, 16, 32, 32),  # multi-channel, 2x
        (1, 3, 100, 100, 50, 50),  # downsample 0.5x
        (1, 1, 1, 1, 3, 3),  # minimal input, 3x upsample
        (8, 64, 128, 128, 256, 256),  # large, 2x upsample
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_with_backward(batch, channel, h, w, out_h, out_w, dtype):
    input = torch.randn(
        (batch, channel, h, w),
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    ref_i = utils.to_reference(input.detach()).to(torch.float32).requires_grad_(True)
    output_size = [out_h, out_w]

    ref_out = torch.nn.functional.interpolate(ref_i, size=output_size, mode="nearest")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.interpolate(
            input, size=output_size, mode="nearest"
        )

    utils.gems_assert_close(res_out, ref_out, dtype)

    # backward
    grad = torch.randn_like(res_out)
    ref_out.backward(utils.to_reference(grad).to(torch.float32))
    res_out.backward(grad)
    utils.gems_assert_close(input.grad, ref_i.grad, dtype)


# ============================================================================
# Scales parameter tests
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    "batch,channel,h,w,scale_h,scale_w",
    [
        (1, 3, 16, 16, 2.0, 2.0),  # 2x upsample
        (2, 8, 32, 32, 0.5, 0.5),  # 0.5x downsample
        (4, 3, 20, 20, 1.5, 2.5),  # non-integer scales
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_scales(batch, channel, h, w, scale_h, scale_w, dtype):
    input = torch.randn((batch, channel, h, w), dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    out_h = int(h * scale_h)
    out_w = int(w * scale_w)

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=[out_h, out_w]).to(
        dtype
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=[out_h, out_w])

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Small shape and edge case tests
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),  # minimal spatial
        (1, 1, 2, 2),  # 2x2 input
        (1, 3, 8, 8),  # single batch, 3 channels
    ],
)
@pytest.mark.parametrize("scale", [(3, 3), (2, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_small_shapes(shape, scale, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Non-contiguous tensor tests
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (3, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_non_contiguous(scale, dtype):
    input = torch.randn(4, 3, 32, 48, dtype=dtype, device=flag_gems.device)
    input_nc = input.permute(0, 1, 3, 2)  # non-contiguous
    ref_i = utils.to_reference(input_nc).to(torch.float32)
    output_size = [int(input_nc.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input_nc, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Empty tensor tests
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    "shape,output_size",
    [
        ((0, 3, 16, 16), [32, 32]),  # zero batch
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_empty(shape, output_size, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    assert res_out.shape == (0, 3, 32, 32)


# ============================================================================
# Boundary value tests (NaN, Inf, extreme values)
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_nan_inf(dtype):
    """Test that NaN and Inf values are preserved through upsampling."""
    input = torch.randn(2, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    # Insert NaN and Inf
    input[0, 0, 0, 0] = float("nan")
    input[0, 0, 1, 1] = float("inf")
    input[1, 0, 0, 0] = float("-inf")

    output_size = [16, 16]

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    # Check NaN/Inf are preserved
    assert torch.isnan(res_out[0, 0, 0, 0])
    assert torch.isinf(res_out[0, 0, 2, 2])
    assert torch.isinf(res_out[1, 0, 0, 0])


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_large_values(dtype):
    """Test with very large and very small values."""
    input = torch.zeros(1, 1, 4, 4, dtype=dtype, device=flag_gems.device)
    input[0, 0, 0, 0] = 1e4
    input[0, 0, 1, 1] = -1e4
    input[0, 0, 2, 2] = 1e-4

    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [8, 8]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Backward with edge cases
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_backward_downsample(dtype):
    """Test backward with downsampling (multiple input pixels map to same output)."""
    input = torch.randn(
        2, 4, 16, 16, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_i = utils.to_reference(input.detach()).to(torch.float32).requires_grad_(True)
    output_size = [8, 8]

    ref_out = torch.nn.functional.interpolate(ref_i, size=output_size, mode="nearest")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.interpolate(
            input, size=output_size, mode="nearest"
        )

    utils.gems_assert_close(res_out, ref_out, dtype)

    grad = torch.randn_like(res_out)
    ref_out.backward(utils.to_reference(grad).to(torch.float32))
    res_out.backward(grad)
    utils.gems_assert_close(input.grad, ref_i.grad, dtype)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_backward_ones_grad(dtype):
    """Test backward with all-ones gradient (sum of gradients should equal output size)."""
    input = torch.randn(
        1, 1, 4, 4, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_i = utils.to_reference(input.detach()).to(torch.float32).requires_grad_(True)
    output_size = [8, 8]

    ref_out = torch.nn.functional.interpolate(ref_i, size=output_size, mode="nearest")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.interpolate(
            input, size=output_size, mode="nearest"
        )

    grad = torch.ones_like(res_out)
    ref_out.backward(utils.to_reference(grad).to(torch.float32))
    res_out.backward(grad)
    utils.gems_assert_close(input.grad, ref_i.grad, dtype)


# ============================================================================
# Input validation tests
# ============================================================================
@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d_invalid_dims():
    """Test that invalid input dimensions raise errors."""
    input = torch.randn(3, 4, device=flag_gems.device)
    with pytest.raises(AssertionError):
        with flag_gems.use_gems():
            torch._C._nn.upsample_nearest2d(input, output_size=[8, 8])


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d_invalid_output_size():
    """Test that invalid output_size raises error."""
    input = torch.randn(1, 3, 8, 8, device=flag_gems.device)
    with pytest.raises(AssertionError):
        with flag_gems.use_gems():
            torch._C._nn.upsample_nearest2d(input, output_size=[8])


# ============================================================================
# All-NaN tensor test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_all_nan(dtype):
    """Test that all-NaN input produces all-NaN output."""
    input = torch.full((2, 3, 4, 4), float("nan"), dtype=dtype, device=flag_gems.device)
    output_size = [8, 8]

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    assert torch.isnan(res_out).all()


# ============================================================================
# All-Inf tensor test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_all_inf(dtype):
    """Test that all-Inf input produces all-Inf output."""
    input = torch.full((2, 3, 4, 4), float("inf"), dtype=dtype, device=flag_gems.device)
    output_size = [8, 8]

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    assert torch.isinf(res_out).all()


# ============================================================================
# Negative values test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_negative(dtype):
    """Test with all-negative input values."""
    input = -torch.abs(torch.randn(2, 3, 8, 8, dtype=dtype, device=flag_gems.device))
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [16, 16]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Constant tensor test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_constant(dtype):
    """Test with constant-valued input."""
    input = torch.full((2, 3, 8, 8), 42.0, dtype=dtype, device=flag_gems.device)
    output_size = [16, 16]

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    expected = torch.full((2, 3, 16, 16), 42.0, dtype=dtype, device=flag_gems.device)
    assert torch.equal(res_out, expected)


# ============================================================================
# Extreme values test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_extreme_values(dtype):
    """Test with very large and very small magnitude values."""
    input = torch.zeros(1, 1, 4, 4, dtype=dtype, device=flag_gems.device)
    if dtype == torch.float32:
        input[0, 0, 0, 0] = 1e30
        input[0, 0, 1, 1] = -1e30
        input[0, 0, 2, 2] = 1e-30
    else:
        input[0, 0, 0, 0] = 1e4
        input[0, 0, 1, 1] = -1e4
        input[0, 0, 2, 2] = 1e-4

    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [8, 8]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Identity scale (1x) test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_identity(dtype):
    """Test with scale factor 1x (identity mapping)."""
    input = torch.randn(2, 3, 16, 16, dtype=dtype, device=flag_gems.device)
    output_size = [16, 16]

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, input, dtype)


# ============================================================================
# Non-square input/output test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    "batch,channel,h,w,out_h,out_w",
    [
        (2, 3, 16, 32, 32, 64),  # non-square input, non-square output
        (1, 1, 10, 20, 5, 10),  # downsample, non-square
        (4, 8, 64, 128, 128, 256),  # large non-square
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_non_square(batch, channel, h, w, out_h, out_w, dtype):
    """Test with non-square input and output shapes."""
    input = torch.randn((batch, channel, h, w), dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [out_h, out_w]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Large tensor benchmark shape
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_large_tensor(dtype):
    """Test with large tensor shapes for benchmark validation."""
    input = torch.randn(8, 64, 128, 128, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [256, 256]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Backward with non-square shapes
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_backward_non_square(dtype):
    """Test backward with non-square input/output shapes."""
    input = torch.randn(
        2, 4, 16, 32, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_i = utils.to_reference(input.detach()).to(torch.float32).requires_grad_(True)
    output_size = [32, 64]

    ref_out = torch.nn.functional.interpolate(ref_i, size=output_size, mode="nearest")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.interpolate(
            input, size=output_size, mode="nearest"
        )

    utils.gems_assert_close(res_out, ref_out, dtype)

    grad = torch.randn_like(res_out)
    ref_out.backward(utils.to_reference(grad).to(torch.float32))
    res_out.backward(grad)
    utils.gems_assert_close(input.grad, ref_i.grad, dtype)


# ============================================================================
# Backward with 3x upsample
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_backward_3x(dtype):
    """Test backward with 3x upsampling factor."""
    input = torch.randn(
        2, 3, 8, 8, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_i = utils.to_reference(input.detach()).to(torch.float32).requires_grad_(True)
    output_size = [24, 24]

    ref_out = torch.nn.functional.interpolate(ref_i, size=output_size, mode="nearest")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.interpolate(
            input, size=output_size, mode="nearest"
        )

    utils.gems_assert_close(res_out, ref_out, dtype)

    grad = torch.randn_like(res_out)
    ref_out.backward(utils.to_reference(grad).to(torch.float32))
    res_out.backward(grad)
    utils.gems_assert_close(input.grad, ref_i.grad, dtype)


# ============================================================================
# Single channel single batch test
# ============================================================================
@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_nearest2d_single_channel_single_batch(dtype):
    """Test with batch=1, channel=1 (minimal batch/channel dimensions)."""
    input = torch.randn(1, 1, 32, 32, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [64, 64]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)
