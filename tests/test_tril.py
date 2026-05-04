import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Original test shapes from FlagGems test framework
SHAPE_DIAGONAL = list(zip(utils.POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = utils.unsqueeze_tensor(inp, 2)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Extended test shapes covering small/medium/large/non-square/multi-dim
TRIL_SHAPES = [
    ((1, 1), 0),  # single element
    ((2, 2), 0),  # small square
    ((8, 8), -1),  # small, negative diagonal
    ((64, 64), 1),  # medium, positive diagonal
    ((128, 256), 0),  # non-square (more cols)
    ((256, 128), 0),  # non-square (more rows)
    ((256, 256), 0),  # medium-large
    ((1024, 1024), 0),  # large
    ((4096, 4096), 0),  # very large
    ((3, 5), 0),  # non-square (more cols)
    ((5, 3), 0),  # non-square (more rows)
    ((1, 5), 0),  # single row
    ((5, 1), 0),  # single col
    ((2, 3, 4), 0),  # 3D
    ((2, 3, 4, 5), -2),  # 4D
    ((2, 3, 4, 5, 6), 1),  # 5D
]

INT_DTYPES = [torch.int32, torch.int64]
BOOL_DTYPES = [torch.bool]


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", TRIL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tril_float_all_shapes(shape, diagonal, dtype):
    """Test tril with all float dtypes across all shape/diagonal combinations."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", TRIL_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_tril_int(shape, diagonal, dtype):
    """Test tril with integer dtypes - must be bit-exact."""
    inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    torch.testing.assert_close(res_out.cpu(), ref_out.cpu())


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", TRIL_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_DTYPES)
def test_tril_bool(shape, diagonal, dtype):
    """Test tril with bool dtype - must be bit-exact."""
    inp = torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    torch.testing.assert_close(res_out.cpu(), ref_out.cpu())


# Test non-contiguous tensors (transposed)
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [((8, 8), 0), ((16, 16), -1), ((32, 32), 1), ((64, 64), 0)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_non_contiguous(shape, diagonal, dtype):
    """Test tril with non-contiguous (transposed) input."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_t = inp.t()

    ref_inp = utils.to_reference(inp_t)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp_t, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Test out parameter
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [((8, 8), 0), ((16, 16), -1), ((32, 32), 1), ((64, 64), 0)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_out(shape, diagonal, dtype):
    """Test tril with explicit out parameter."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        torch.tril(inp, diagonal, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


# Test edge cases
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [
        ((1, 1), 0),
        ((1, 5), 0),
        ((5, 1), 0),
        ((0, 5), 0),
        ((5, 0), 0),
        ((1, 1), 1),
        ((1, 1), -1),
    ],
)
def test_tril_edge_cases(shape, diagonal):
    """Test tril with edge case shapes."""
    inp = torch.randn(shape, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# Test all diagonal values
@pytest.mark.tril
@pytest.mark.parametrize("diagonal", [-3, -2, -1, 0, 1, 2, 3])
def test_tril_diagonals(diagonal):
    """Test tril with all diagonal values on a fixed-size matrix."""
    shape = (8, 8)
    inp = torch.randn(shape, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# Test in-place tril_
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [((8, 8), 0), ((16, 16), -1), ((32, 32), 1), ((2, 3, 4), 0)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_inplace(shape, diagonal, dtype):
    """Test in-place tril_ operation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        inp.tril_(diagonal)

    utils.gems_assert_close(inp, ref_out, dtype)


# Test NaN/Inf handling
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [((8, 8), 0), ((16, 16), -1), ((32, 32), 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_nan_inf(shape, diagonal, dtype):
    """Test tril correctly handles NaN and Inf values.

    tril is a selection operation: elements below/on diagonal are kept as-is
    (including NaN/Inf), elements above diagonal become 0.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Inject NaN and Inf at known positions
    inp[0, 0] = float("nan")
    inp[0, 1] = float("inf")
    inp[1, 0] = float("-inf")
    inp[1, 1] = float("nan")
    if shape[0] > 2 and shape[1] > 2:
        inp[2, 0] = float("inf")
        inp[2, 1] = float("-inf")
        inp[2, 2] = float("nan")

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    # Compare element-by-element, handling NaN separately
    res_cpu = res_out.cpu()
    ref_cpu = ref_out.cpu()

    # Check that NaN positions match
    nan_mask = torch.isnan(ref_cpu)
    assert torch.isnan(res_cpu[nan_mask]).all(), "NaN positions don't match"
    assert not torch.isnan(res_cpu[~nan_mask]).any(), "Unexpected NaN in output"

    # Check non-NaN values are close (use equal_nan=True to handle NaN)
    flag_gems.testing.assert_close(res_out, ref_out, dtype, equal_nan=True)


# Test all-NaN and all-Inf tensors
@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", [((8, 8), 0), ((16, 16), -1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_all_nan(shape, diagonal, dtype):
    """Test tril with all-NaN input tensor."""
    inp = torch.full(shape, float("nan"), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    # All values should be NaN (below diagonal) or 0 (above diagonal)
    res_cpu = res_out.cpu()
    ref_cpu = ref_out.cpu()
    nan_mask = torch.isnan(ref_cpu)
    assert torch.isnan(res_cpu[nan_mask]).all(), "NaN positions don't match"
    assert (res_cpu[~nan_mask] == 0).all(), "Above-diagonal zeros missing"


@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", [((8, 8), 0), ((16, 16), 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_all_inf(shape, diagonal, dtype):
    """Test tril with all-Inf input tensor."""
    inp = torch.full(shape, float("inf"), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Test extreme float values
@pytest.mark.tril
@pytest.mark.parametrize(
    "shape, diagonal",
    [((8, 8), 0), ((16, 16), -1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_extreme_values(shape, diagonal, dtype):
    """Test tril with very large and very small float values."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.float32:
        inp[0, 0] = 1e30
        inp[0, 1] = -1e30
        inp[1, 0] = 1e-30
    else:
        inp[0, 0] = 1e4
        inp[0, 1] = -1e4
        inp[1, 0] = 1e-4

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Test all-same-value tensor
@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", [((8, 8), 0), ((16, 16), 1), ((4, 4), -1)])
def test_tril_constant(shape, diagonal):
    """Test tril with all elements equal to same value."""
    inp = torch.full(shape, 7.0, device=flag_gems.device)

    ref_out = torch.tril(inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# Test negative values
@pytest.mark.tril
@pytest.mark.parametrize("shape, diagonal", [((8, 8), 0), ((16, 16), -1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tril_negative(shape, diagonal, dtype):
    """Test tril with all-negative input."""
    inp = -torch.abs(torch.randn(shape, dtype=dtype, device=flag_gems.device))

    ref_inp = utils.to_reference(inp)
    ref_out = torch.tril(ref_inp, diagonal)

    with flag_gems.use_gems():
        res_out = torch.tril(inp, diagonal)

    utils.gems_assert_close(res_out, ref_out, dtype)
