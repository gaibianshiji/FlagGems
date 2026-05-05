"""
Comprehensive test suite for CTC loss operator.

Coverage:
- Input sizes: tiny (3,1,2) through xlarge (500,128,50), 15 shape configurations
- Reduction modes: none, mean, sum + consistency cross-checks
- Data types: float32, float16, bfloat16
- Edge cases: target_length=0, all blank, extreme values, target==input length,
  blank!=0, non-contiguous, NaN, -Inf, repeated targets, single char, long targets,
  variable input_lengths, small/large vocabulary, batch=1, large batch
- Dispatch: use_gems() context (float32), direct flag_gems.ctc_loss (all dtypes)

Note: PyTorch's torch.ctc_loss C++ builtin rejects float16/bfloat16 before
dispatching to the aten op. Our aten override works for float32 via use_gems().
For float16/bfloat16, we call flag_gems.ctc_loss directly.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# === Shape definitions: 15 configurations from tiny to xlarge ===

_CTC_SHAPES_TINY = [(3, 1, 2), (2, 1, 3)]  # minimal
_CTC_SHAPES_SMALL = [(5, 1, 3), (10, 2, 5), (8, 4, 8)]  # small
_CTC_SHAPES_MEDIUM = [(50, 8, 20), (100, 16, 26), (80, 32, 30)]  # medium
_CTC_SHAPES_LARGE = [(200, 32, 50), (100, 64, 100), (500, 16, 50)]  # large
_CTC_SHAPES_XLARGE = [(500, 64, 50), (200, 128, 50)]  # xlarge

_CTC_SHAPES = (
    [(5, 1, 3)]
    if QUICK_MODE
    else (_CTC_SHAPES_TINY + _CTC_SHAPES_SMALL + _CTC_SHAPES_MEDIUM + _CTC_SHAPES_LARGE + _CTC_SHAPES_XLARGE)
)

_CTC_DTYPES = [torch.float32] if QUICK_MODE else [torch.float32, torch.float16, torch.bfloat16]
_CTC_DTYPES_FP16 = [torch.float32] if QUICK_MODE else [torch.float32, torch.float16]


def _ctc_loss_ref(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """Reference CTC loss on CPU with float32.

    PyTorch's CTC loss doesn't support float16/bfloat16 on CPU or CUDA,
    so we cast to float32 and compute on CPU for reference.
    Returns result on the same device as input log_probs.
    """
    ref_lp = log_probs.detach().cpu().float()
    ref_t = targets.detach().cpu()
    ref_il = input_lengths.detach().cpu()
    ref_tl = target_lengths.detach().cpu()
    result = torch.nn.functional.ctc_loss(
        ref_lp,
        ref_t,
        ref_il,
        ref_tl,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return result.to(device=log_probs.device, dtype=log_probs.dtype)


def _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, **kwargs):
    """Helper to run CTC loss test with appropriate dispatch method."""
    reduction = kwargs.get("reduction", "mean")
    ref_out = _ctc_loss_ref(log_probs, targets, input_lengths, target_lengths, **kwargs)

    if dtype == torch.float32:
        with flag_gems.use_gems():
            res_out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, **kwargs)
    else:
        reduction_map = {"none": 0, "mean": 1, "sum": 2}
        kw = {k: v for k, v in kwargs.items() if k != "reduction"}
        kw["reduction"] = reduction_map.get(reduction, 1)
        res_out = flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, **kw)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================
# 1. Main parametrized test: all shapes × reductions × dtypes
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T,B,C", _CTC_SHAPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("dtype", _CTC_DTYPES)
def test_ctc_loss(T, B, C, reduction, dtype):
    """Test CTC loss with various input sizes, reduction modes, and dtypes."""
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction=reduction)


# ============================================================
# 2. Parameter-specific tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("blank", [0, 1, 5])
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_blank_values(blank, dtype):
    """Test CTC loss with different blank label positions."""
    T, B, C = 20, 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(0, C, (B, 8), device=flag_gems.device)
    # Ensure targets don't equal blank
    targets = targets.where(targets != blank, (targets + 1) % C)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 8, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, blank=blank, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("zero_infinity", [True, False])
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_zero_infinity(zero_infinity, dtype):
    """Test CTC loss with zero_infinity parameter."""
    T, B, C = 15, 3, 8
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 6), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 6, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype,
        reduction="mean",
        zero_infinity=zero_infinity,
    )


# ============================================================
# 3. Reduction consistency tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T,B,C", [(50, 8, 20), (100, 16, 50)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_reduction_consistency(T, B, C, dtype):
    """Verify: sum(none) == sum, mean(none) == mean (with target_lengths)."""
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 8)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)

    r_none = flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=0)
    r_sum = flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=2)
    r_mean = flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction=1)

    # sum(none) == sum
    assert torch.allclose(r_none.sum(), r_sum, rtol=1e-4, atol=1e-5), f"sum(none)={r_none.sum()}, sum={r_sum}"
    # mean vs PyTorch reference
    ref_mean = _ctc_loss_ref(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    assert torch.allclose(r_mean, ref_mean, rtol=1e-4, atol=1e-5), f"mean={r_mean}, ref={ref_mean}"


# ============================================================
# 4. Variable length tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_variable_input_lengths(dtype):
    """Test with varying input_lengths across batch (some shorter than T)."""
    T, B, C = 30, 6, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 8), device=flag_gems.device)
    input_lengths = torch.tensor([5, 10, 15, 20, 25, 30], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([3, 5, 5, 6, 7, 8], device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_variable_target_lengths(dtype):
    """Test with varying target_lengths across batch."""
    T, B, C = 20, 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 10), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([2, 5, 8, 10], device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_mixed_lengths(dtype):
    """Test with both input_lengths and target_lengths varying."""
    T, B, C = 50, 8, 20
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    input_lengths = torch.tensor([10, 20, 30, 40, 50, 10, 20, 30], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([3, 5, 8, 10, 5, 3, 5, 8], device=flag_gems.device, dtype=torch.long)
    max_s = target_lengths.max().item()
    targets = torch.randint(1, C, (B, max_s), device=flag_gems.device)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 5. Vocabulary size tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("C", [2, 3, 26, 50, 100, 200])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_vocabulary_sizes(C, dtype):
    """Test with various vocabulary sizes (number of classes)."""
    T, B = 50, 8
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 6. Batch size tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("B", [1, 2, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_batch_sizes(B, dtype):
    """Test with various batch sizes including B=1 and large batches."""
    T, C = 50, 26
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 7. Time dimension tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T", [1, 2, 5, 50, 200, 500, 1000])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_time_steps(T, dtype):
    """Test with various time dimensions including T=1 (minimal) and T=1000 (long)."""
    B, C = 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 5)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 8. Target pattern tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_single_char_target(dtype):
    """Test with single-character target (S=1)."""
    T, B, C = 20, 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 1), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 1, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_repeated_target(dtype):
    """Test with repeated characters in target (e.g., 'aaaa')."""
    T, B, C = 30, 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    # All same character
    char = torch.randint(1, C, (1,)).item()
    targets = torch.full((B, 5), char, device=flag_gems.device, dtype=torch.long)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 5, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_long_target(dtype):
    """Test with target length close to input length."""
    T, B, C = 30, 4, 50
    S = T - 5  # target almost as long as input
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_all_same_target(dtype):
    """Test where every batch element has the same target sequence."""
    T, B, C = 20, 8, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    single_target = torch.tensor([3, 5, 7], device=flag_gems.device)
    targets = single_target.unsqueeze(0).expand(B, -1).contiguous()
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 3, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 9. Edge cases
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_target_length_zero(dtype):
    """Test CTC loss with target_length=0 (empty target)."""
    T, B, C = 10, 2, 5
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 5), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([0, 3], device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_all_blank_timesteps(dtype):
    """Test CTC loss where all timesteps predict blank (high loss expected)."""
    T, B, C = 8, 2, 5
    log_probs = torch.full((T, B, C), -10.0, dtype=dtype, device=flag_gems.device)
    log_probs[:, :, 0] = 0.0
    log_probs = log_probs.log_softmax(2)
    targets = torch.randint(1, C, (B, 3), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 3, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_target_equals_input_length(dtype):
    """Test CTC loss where target length equals input length (no room for blanks)."""
    T, B, C = 10, 2, 8
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, T), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_special_values(dtype):
    """Test CTC loss with extreme log probability values."""
    T, B, C = 5, 2, 4
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    extreme_val = -1e4 if dtype == torch.float16 else -1e6
    log_probs[0, 0, 0] = extreme_val
    log_probs[1, 1, 1] = 0.0
    targets = torch.randint(1, C, (B, 3), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 3, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_blank_nonzero(dtype):
    """Test CTC loss with blank label not at index 0."""
    T, B, C = 15, 3, 8
    blank = C - 1  # blank = last class
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(0, blank, (B, 6), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 6, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, blank=blank, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_noncontiguous(dtype):
    """Test CTC loss with non-contiguous log_probs (operator calls .contiguous() internally)."""
    T, B, C = 20, 4, 10
    raw = torch.randn(T, B, C * 2, dtype=dtype, device=flag_gems.device)
    log_probs_nc = raw[:, :, ::2]  # stride-2 on last dim → non-contiguous
    log_probs = torch.nn.functional.log_softmax(log_probs_nc, dim=2)
    targets = torch.randint(1, C, (B, 8), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 8, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_nan_input(dtype):
    """Test CTC loss with NaN values in log_probs — should propagate NaN."""
    T, B, C = 10, 2, 5
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    log_probs[3, 0, 2] = float("nan")
    log_probs[7, 1, 1] = float("nan")
    targets = torch.randint(1, C, (B, 4), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 4, device=flag_gems.device, dtype=torch.long)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    assert torch.isnan(res_out), f"Expected NaN output, got {res_out}"


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_inf_input(dtype):
    """Test CTC loss with -Inf values in log_probs (valid in log domain)."""
    T, B, C = 10, 2, 5
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    log_probs[0, 0, 1] = float("-inf")
    log_probs[5, 1, 3] = float("-inf")
    targets = torch.randint(1, C, (B, 4), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 4, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_all_inf_log_probs(dtype):
    """Test CTC loss where all log_probs are -Inf (degenerate case)."""
    T, B, C = 5, 2, 3
    log_probs = torch.full((T, B, C), float("-inf"), dtype=dtype, device=flag_gems.device)
    # Give one valid entry per timestep to avoid all-NaN
    for t in range(T):
        log_probs[t, :, 0] = 0.0
    targets = torch.randint(1, C, (B, 2), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 2, device=flag_gems.device, dtype=torch.long)

    ref_out = _ctc_loss_ref(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    # Should be inf (very high loss since targets can't be produced)
    assert (
        torch.isinf(res_out) or torch.isnan(res_out) or torch.allclose(res_out, ref_out, rtol=1e-3, atol=1.0)
    ), f"Unexpected result: {res_out}, ref: {ref_out}"


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_short_input_long_target(dtype):
    """Test where input_length is barely longer than target_length."""
    T, B, C = 10, 4, 10
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 8), device=flag_gems.device)
    input_lengths = torch.tensor([9, 9, 10, 10], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([8, 8, 8, 8], device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_input_shorter_than_target(dtype):
    """Test where some input_lengths < target_lengths (invalid but should handle gracefully)."""
    T, B, C = 10, 3, 5
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 5), device=flag_gems.device)
    input_lengths = torch.tensor([5, 10, 3], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([3, 3, 5], device=flag_gems.device, dtype=torch.long)
    # Third sample has input_length=3 < target_length=5 — loss should be very high or inf
    ref_out = _ctc_loss_ref(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    with flag_gems.use_gems():
        res_out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
    # Both should produce similar results (likely inf)
    assert (
        torch.isinf(res_out) or torch.isnan(res_out) or torch.allclose(res_out, ref_out, rtol=1e-3, atol=1.0)
    ), f"Mismatch: gems={res_out}, ref={ref_out}"


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_zero_input_length(dtype):
    """Test with zero input_length for some batch elements."""
    T, B, C = 10, 3, 5
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    targets = torch.randint(1, C, (B, 3), device=flag_gems.device)
    input_lengths = torch.tensor([0, 5, 10], device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.tensor([0, 3, 3], device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_uniform_log_probs(dtype):
    """Test with near-uniform log probabilities (model has no preference)."""
    T, B, C = 20, 4, 10
    log_probs = torch.full((T, B, C), -torch.log(torch.tensor(float(C))), dtype=dtype, device=flag_gems.device)
    targets = torch.randint(1, C, (B, 5), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), 5, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", _CTC_DTYPES_FP16)
def test_ctc_loss_high_confidence(dtype):
    """Test with very peaked log probabilities (model very confident)."""
    T, B, C = 20, 4, 10
    log_probs = torch.full((T, B, C), -100.0, dtype=dtype, device=flag_gems.device)
    S = 5
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    # Make the target characters very likely
    for b in range(B):
        for t in range(min(T, S)):
            log_probs[t, b, targets[b, t]] = 0.0
    log_probs = torch.nn.functional.log_softmax(log_probs.float(), dim=2).to(dtype)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 10. dtype-specific stress tests
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T,B,C", [(100, 32, 50), (200, 64, 50)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_ctc_loss_fp16_large(dtype, T, B, C):
    """Stress test float16 with large shapes."""
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


@pytest.mark.ctc_loss
@pytest.mark.parametrize("T,B,C", [(100, 32, 50), (200, 64, 50)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ctc_loss_bf16_large(dtype, T, B, C):
    """Stress test bfloat16 with large shapes."""
    log_probs = torch.randn(T, B, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    S = min(C - 1, 10)
    targets = torch.randint(1, C, (B, S), device=flag_gems.device)
    input_lengths = torch.full((B,), T, device=flag_gems.device, dtype=torch.long)
    target_lengths = torch.full((B,), S, device=flag_gems.device, dtype=torch.long)
    _run_ctc_test(log_probs, targets, input_lengths, target_lengths, dtype, reduction="mean")


# ============================================================
# 11. Cross-shape numerical comparison
# ============================================================


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ctc_loss_large_small_consistency(dtype):
    """Verify small shape result is consistent when embedded in large batch."""
    T, C = 20, 10
    # Small: B=1
    lp_small = torch.randn(T, 1, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    tgt_small = torch.randint(1, C, (1, 5), device=flag_gems.device)
    il_small = torch.full((1,), T, device=flag_gems.device, dtype=torch.long)
    tl_small = torch.full((1,), 5, device=flag_gems.device, dtype=torch.long)

    r_small = flag_gems.ctc_loss(lp_small, tgt_small, il_small, tl_small, reduction=0)

    # Large: B=32, with same first element
    lp_large = torch.randn(T, 32, C, dtype=dtype, device=flag_gems.device).log_softmax(2)
    lp_large[:, 0, :] = lp_small[:, 0, :]
    tgt_large = torch.randint(1, C, (32, 5), device=flag_gems.device)
    tgt_large[0, :] = tgt_small[0, :]
    il_large = torch.full((32,), T, device=flag_gems.device, dtype=torch.long)
    tl_large = torch.full((32,), 5, device=flag_gems.device, dtype=torch.long)

    r_large = flag_gems.ctc_loss(lp_large, tgt_large, il_large, tl_large, reduction=0)

    assert torch.allclose(
        r_small[0], r_large[0], rtol=1e-5, atol=1e-6
    ), f"Small: {r_small[0]}, Large-batch[0]: {r_large[0]}"
