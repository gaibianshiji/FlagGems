import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SHAPES = [(32, 8)]
    DIMS = [0, 1]
else:
    FLOAT_DTYPES = [torch.float16, torch.float32]
    SHAPES = [(128, 16), (256, 32), (512, 128)]
    DIMS = [0, 1]

random.seed(time.time() // 100)


def _make_unique_index(index, dim, size_dim):
    """Fill index with unique random values along `dim` to avoid contention."""
    shape = list(index.shape)
    m, n = shape[0], shape[1]
    index_size_dim = shape[dim]
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            ii = [i, j]
            ii[dim] = slice(0, index.size(dim) + 1)
            index[tuple(ii)] = torch.randperm(size_dim, device=index.device)[
                :index_size_dim
            ]


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_two(shape, dim, dtype, reduce):
    utils.init_seed(0)
    device = flag_gems.device

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    size_dim = min(shape[dim], shape[dim])

    index = torch.empty(shape, dtype=torch.long, device=device)
    _make_unique_index(index, dim, size_dim)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce)

    if reduce in ("amax", "amin"):
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_two_include_self_false(shape, dim, dtype, reduce):
    utils.init_seed(0)
    device = flag_gems.device

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    size_dim = shape[dim]

    index = torch.empty(shape, dtype=torch.long, device=device)
    _make_unique_index(index, dim, size_dim)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=False
    )

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce, include_self=False)

    if reduce in ("amax", "amin"):
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64, 32), (16, 8, 4, 2), (8, 4, 2, 2, 2)],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduce", ["sum", "amax"])
def test_scatter_reduce_two_ndim(shape, dtype, reduce):
    """Test 1D, 3D, 4D, 5D tensors to exercise 5D coordinate decoding."""
    utils.init_seed(0)
    device = flag_gems.device

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    dim = len(shape) - 1
    size_dim = shape[dim]

    index = torch.empty(shape, dtype=torch.long, device=device)
    flat_idx = torch.randint(0, size_dim, (index.numel(),), device=device)
    index.copy_(flat_idx.reshape(shape))

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce)

    if reduce in ("amax", "amin"):
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_two_contention(dtype):
    """Test contention: all indices point to the same output element."""
    utils.init_seed(0)
    device = flag_gems.device
    shape = (64, 64)
    dim = 1

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    index = torch.zeros(shape, dtype=torch.long, device=device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, "sum")

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, "sum")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_two_large(dtype):
    """Test large tensors to verify int64 offset handling."""
    utils.init_seed(0)
    device = flag_gems.device
    shape = (4096, 4096)
    dim = 1

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    index = torch.empty(shape, dtype=torch.long, device=device)
    _make_unique_index(index, dim, shape[dim])

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, "sum")

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, "sum")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_two_(shape, dim, dtype, reduce):
    """In-place variant test."""
    utils.init_seed(0)
    device = flag_gems.device

    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    size_dim = shape[dim]

    index = torch.empty(shape, dtype=torch.long, device=device)
    _make_unique_index(index, dim, size_dim)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        inp.scatter_reduce_(dim, index, src, reduce)

    if reduce in ("amax", "amin"):
        utils.gems_assert_equal(inp, ref_out)
    else:
        utils.gems_assert_close(inp, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape", [(64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduce", ["sum", "amax"])
def test_scatter_reduce_two_noncontiguous(shape, dtype, reduce):
    """Test with non-contiguous input via transpose."""
    utils.init_seed(0)
    device = flag_gems.device

    inp_t = torch.randn((shape[1], shape[0]), dtype=dtype, device=device)
    inp = inp_t.t()  # non-contiguous
    src_t = torch.randn((shape[1], shape[0]), dtype=dtype, device=device)
    src = src_t.t()

    dim = 0
    index = torch.empty(shape, dtype=torch.long, device=device)
    _make_unique_index(index, dim, shape[dim])

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce)

    if reduce in ("amax", "amin"):
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
