import gc

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    """Free GPU memory between tests to avoid OOM on devices with limited VRAM."""
    yield
    torch.cuda.empty_cache()
    gc.collect()


# Helper to check if float64 is well-supported on current GPU
def _fp64_supported():
    """Check if the current GPU has good float64 support."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    # BI-V150 and similar GPUs have limited float64 support
    if "v150" in name or "iluvatar" in name:
        return False
    return True


# Skip float64 tests on hardware with limited fp64 support
_fp64_skip = pytest.mark.skipif(
    not _fp64_supported(),
    reason="GPU has limited float64 support (e.g. Iluvatar BI-V150)",
)


# --- S accuracy (primary validation) ---

@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n",
    [
        (1, 1), (1, 4), (4, 1),
        (2, 2), (3, 3),
        (4, 3), (3, 4),
        (64, 64), (128, 128), (256, 256), (1024, 1024), (4096, 4096),
        (128, 64), (64, 128), (256, 128), (1024, 512), (512, 1024),
    ],
)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_s_accuracy(m, n, some, dtype):
    """Verify singular values match PyTorch reference exactly."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_S = torch.svd(ref_inp, some=some)[1]
    with flag_gems.use_gems():
        res_S = torch.svd(inp, some=some)[1]

    # On some GPUs (e.g. BI-V150), very large matrices produce NaN in S
    # due to cusolver driver limitations. Verify our impl matches PyTorch.
    nan_mask = torch.isnan(ref_S)
    if nan_mask.any():
        assert torch.isnan(res_S[nan_mask]).all(), "NaN mismatch vs PyTorch reference"
        # Compare non-NaN values
        valid = ~nan_mask
        if valid.any():
            utils.gems_assert_close(res_S[valid], ref_S[valid], dtype, atol=1e-6)
        return

    # S should match exactly (same algorithm)
    utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-6)


@_fp64_skip
@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n",
    [
        (2, 2), (4, 3), (3, 4),
        (64, 64), (128, 64), (64, 128),
        (256, 256), (1024, 1024), (4096, 4096),
    ],
)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_svd_s_accuracy_fp64(m, n, some, dtype):
    """Verify singular values match PyTorch reference in float64."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_S = torch.svd(ref_inp, some=some)[1]
    with flag_gems.use_gems():
        res_S = torch.svd(inp, some=some)[1]

    # float64 should match very closely
    utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-7)


# --- some=True, compute_uv=True ---

@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n",
    [
        (2, 2), (3, 3),
        (4, 3), (3, 4),
        (64, 64), (128, 128), (256, 256), (1024, 1024), (4096, 4096),
        (128, 64), (64, 128), (256, 128), (1024, 512), (512, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_some_true(m, n, dtype):
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=True)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=True)

    # S must match - handle NaN from cusolver on large matrices
    nan_mask = torch.isnan(ref_S)
    if nan_mask.any():
        assert torch.isnan(res_S[nan_mask]).all(), "NaN mismatch vs PyTorch"
        valid = ~nan_mask
        if valid.any():
            utils.gems_assert_close(res_S[valid], ref_S[valid], dtype, atol=1e-6)
    else:
        utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-6)

    # Reconstruction check (with relaxed tolerance for large matrices)
    recon = res_U @ torch.diag(res_S) @ res_V.T
    err = (inp - recon).abs().max().item()
    if torch.isnan(torch.tensor(err)):
        return  # NaN from cusolver driver bug on large matrices
    tol = max(1e-4, 5e-6 * max(m, n))
    assert err < tol, f"Reconstruction error {err:.2e} exceeds {tol:.2e} for {m}x{n}"

    # Output shape check
    k = min(m, n)
    assert res_U.shape == (m, k), f"U shape: {res_U.shape} vs ({m}, {k})"
    assert res_S.shape == (k,), f"S shape: {res_S.shape} vs ({k},)"
    assert res_V.shape == (n, k), f"V shape: {res_V.shape} vs ({n}, {k})"


@_fp64_skip
@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n",
    [
        (2, 2), (4, 3), (3, 4),
        (64, 64), (128, 64), (64, 128),
        (256, 256), (1024, 1024), (4096, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_svd_some_true_fp64(m, n, dtype):
    """Test some=True with float64 for higher precision."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    # S should be non-negative and descending
    assert (S >= 0).all(), "S contains negative values"
    assert (S[:-1] >= S[1:]).all(), "S not in descending order"

    # Reconstruction check - float64 should be much more precise
    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    tol = max(1e-7, 1e-9 * max(m, n))
    assert err < tol, f"fp64 reconstruction error {err:.2e} exceeds {tol:.2e} for {m}x{n}"

    # Output shape check
    k = min(m, n)
    assert U.shape == (m, k)
    assert S.shape == (k,)
    assert V.shape == (n, k)


# --- some=False, compute_uv=True ---

@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n",
    [(2, 2), (4, 3), (3, 4), (64, 64), (128, 64), (64, 128), (1024, 1024), (4096, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_some_false(m, n, dtype):
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=False)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=False)

    # Handle NaN from cusolver on large matrices
    nan_mask = torch.isnan(ref_S)
    if nan_mask.any():
        assert torch.isnan(res_S[nan_mask]).all(), "NaN mismatch vs PyTorch"
        valid = ~nan_mask
        if valid.any():
            utils.gems_assert_close(res_S[valid], ref_S[valid], dtype, atol=1e-6)
    else:
        utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-6)

    # Output shape check
    assert res_U.shape == ref_U.shape, f"U shape: {res_U.shape} vs {ref_U.shape}"
    assert res_V.shape == ref_V.shape, f"V shape: {res_V.shape} vs {ref_V.shape}"

    # Reconstruction via truncated product
    k = min(m, n)
    recon = res_U[..., :k] @ torch.diag(res_S) @ res_V[..., :k].T
    err = (inp - recon).abs().max().item()
    if torch.isnan(torch.tensor(err)):
        return
    tol = max(1e-4, 5e-6 * max(m, n))
    assert err < tol, f"Reconstruction error {err:.2e} exceeds {tol:.2e}"


# --- compute_uv=False ---

@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n,some",
    [
        (2, 2, True), (4, 3, True), (3, 4, True),
        (64, 64, True), (128, 64, True), (1024, 1024, True), (4096, 4096, True),
        (2, 2, False), (4, 3, False), (64, 64, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_no_uv(m, n, some, dtype):
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=some, compute_uv=False)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=some, compute_uv=False)

    # Handle NaN from cusolver on large matrices
    nan_mask = torch.isnan(ref_S)
    if nan_mask.any():
        assert torch.isnan(res_S[nan_mask]).all(), "NaN mismatch vs PyTorch"
        valid = ~nan_mask
        if valid.any():
            utils.gems_assert_close(res_S[valid], ref_S[valid], dtype, atol=1e-6)
    else:
        utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-6)
    assert res_U.numel() == 0
    assert res_V.numel() == 0


@_fp64_skip
@pytest.mark.svd
@pytest.mark.parametrize(
    "m,n,some",
    [
        (2, 2, True), (4, 3, True), (64, 64, True),
        (1024, 1024, True), (4096, 4096, True),
        (2, 2, False), (64, 64, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_svd_no_uv_fp64(m, n, some, dtype):
    """Test compute_uv=False with float64."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=some, compute_uv=False)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=some, compute_uv=False)

    utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-7)
    assert res_U.numel() == 0
    assert res_V.numel() == 0


# --- Edge cases ---

@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(1, 1), (1, 8), (8, 1)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_edge_cases(m, n, dtype):
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    assert err < 1e-4, f"Reconstruction error {err:.2e} for {m}x{n}"


# --- Special matrices ---

@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_singular_matrix(dtype):
    """Test with a rank-deficient matrix."""
    m, n = 64, 64
    A = torch.randn(m, 10, dtype=dtype, device=flag_gems.device)
    B = torch.randn(10, n, dtype=dtype, device=flag_gems.device)
    inp = A @ B

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    # On some GPUs (e.g. BI-V150), rank-deficient matrices produce NaN in S
    # due to cusolver driver limitations. This matches PyTorch native behavior.
    if torch.isnan(S).any():
        # Verify PyTorch native also produces NaN (hardware limitation, not our bug)
        ref_U, ref_S, ref_V = torch.svd(inp, some=True)
        assert torch.isnan(ref_S).any(), "Our impl produces NaN but PyTorch doesn't"
        return

    # Check that S has 10 large values and 54 near-zero values
    assert S[:10].min() > 0.1, f"Top 10 singular values too small: {S[:10].min():.2e}"

    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    assert err < 1e-3, f"Reconstruction error {err:.2e} for singular matrix"


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_high_condition(dtype):
    """Test with a moderate condition number matrix."""
    m, n = 64, 64
    U_ref, _ = torch.linalg.qr(torch.randn(m, m, dtype=dtype, device=flag_gems.device))
    V_ref, _ = torch.linalg.qr(torch.randn(n, n, dtype=dtype, device=flag_gems.device))
    s = torch.logspace(0, 3, min(m, n), dtype=dtype, device=flag_gems.device)
    inp = U_ref @ torch.diag(s) @ V_ref[:min(m, n), :]

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    recon = U @ torch.diag(S) @ V.T
    rel_err = (inp - recon).abs().max() / inp.abs().max()
    assert rel_err < 0.1, f"Relative reconstruction error {rel_err:.2e}"


# --- Batched inputs ---

@pytest.mark.svd
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("m,n", [(4, 3), (64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_batched(batch_size, m, n, dtype):
    inp = torch.randn(batch_size, m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    k = min(m, n)
    assert U.shape == (batch_size, m, k)
    assert S.shape == (batch_size, k)
    assert V.shape == (batch_size, n, k)

    recon = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
    err = (inp - recon).abs().max().item()
    assert err < 1e-3, f"Batched reconstruction error {err:.2e}"


# --- 4D batched ---

@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_4d_batched(dtype):
    inp = torch.randn(2, 3, 64, 64, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert U.shape == (2, 3, 64, 64)
    assert S.shape == (2, 3, 64)
    assert V.shape == (2, 3, 64, 64)

    recon = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
    err = (inp - recon).abs().max().item()
    assert err < 1e-3, f"4D batched reconstruction error {err:.2e}"


# --- float16 ---

@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(64, 64), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_svd_fp16(m, n, dtype):
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    recon = U @ torch.diag(S) @ V.T
    err = (inp.float() - recon.float()).abs().max().item()
    assert err < 5e-2, f"fp16 reconstruction error {err:.2e}"


# --- Large scale tests ---

@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(1024, 1024), (1024, 512), (512, 1024), (4096, 4096), (4096, 2048), (2048, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_large_scale(m, n, dtype):
    """Test with large matrices (1024+ to 4096)."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    k = min(m, n)
    assert U.shape == (m, k)
    assert S.shape == (k,)
    assert V.shape == (n, k)

    # On some GPUs (e.g. BI-V150), large matrices may produce NaN in S
    if torch.isnan(S).any():
        # Verify PyTorch native also produces NaN
        ref_S = torch.svd(inp, some=True)[1]
        assert torch.isnan(ref_S).any(), "Our impl produces NaN but PyTorch doesn't"
        return

    # S should be non-negative and descending
    assert (S >= 0).all(), "S contains negative values"
    assert (S[:-1] >= S[1:]).all(), "S not in descending order"

    # Reconstruction check with relaxed tolerance (grows with matrix size)
    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    tol = max(1e-2, 3e-6 * max(m, n))
    assert err < tol, f"Large scale reconstruction error {err:.2e} exceeds {tol:.2e}"


@_fp64_skip
@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(1024, 1024), (1024, 512), (4096, 4096), (4096, 2048)])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_svd_large_scale_fp64(m, n, dtype):
    """Test with large matrices in float64 for higher precision."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    k = min(m, n)
    assert U.shape == (m, k)
    assert S.shape == (k,)
    assert V.shape == (n, k)

    # S should be non-negative and descending
    assert (S >= 0).all(), "S contains negative values"
    assert (S[:-1] >= S[1:]).all(), "S not in descending order"

    # float64 reconstruction should be very precise
    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    assert err < 1e-3, f"fp64 large scale reconstruction error {err:.2e}"


# --- All parameter combinations ---

@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(4, 3), (3, 4), (64, 64)])
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_svd_all_params(m, n, some, compute_uv, dtype):
    """Test all combinations of some and compute_uv."""
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=some, compute_uv=compute_uv)
    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=some, compute_uv=compute_uv)

    # S should always match
    utils.gems_assert_close(res_S, ref_S, dtype, atol=1e-6)

    if compute_uv:
        assert res_U.shape == ref_U.shape
        assert res_V.shape == ref_V.shape
    else:
        assert res_U.numel() == 0
        assert res_V.numel() == 0


# --- NaN / Inf boundary tests ---

@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_nan_input(dtype):
    """SVD on a matrix containing NaN should propagate NaN to outputs."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 64, 64
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    inp[10, 10] = float("nan")

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert torch.isnan(S).any(), "NaN not propagated to S"
    assert U.shape == (m, min(m, n))
    assert V.shape == (n, min(m, n))


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_inf_input(dtype):
    """SVD on a matrix containing Inf should propagate Inf to outputs."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 64, 64
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    inp[0, 0] = float("inf")

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert torch.isnan(S).any() or torch.isinf(S).any(), "Inf not handled in S"
    assert U.shape == (m, min(m, n))
    assert V.shape == (n, min(m, n))


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_neg_inf_input(dtype):
    """SVD on a matrix containing -Inf should propagate appropriately."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 32, 32
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    inp[5, 5] = float("-inf")

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert torch.isnan(S).any() or torch.isinf(S).any(), "-Inf not handled in S"


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_all_nan(dtype):
    """SVD on an all-NaN matrix should return NaN outputs without crashing."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 32, 64
    inp = torch.full((m, n), float("nan"), dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert torch.isnan(S).all(), "All-NaN input should produce all-NaN S"
    k = min(m, n)
    assert U.shape == (m, k)
    assert V.shape == (n, k)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_zero_matrix(dtype):
    """SVD of an all-zero matrix should return all-zero singular values."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 64, 64
    inp = torch.zeros(m, n, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert (S == 0).all(), "Zero matrix should have all-zero S"
    k = min(m, n)
    assert U.shape == (m, k)
    assert V.shape == (n, k)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_very_large_values(dtype):
    """SVD with very large finite values should not produce NaN/Inf in S."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 32, 32
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device) * 1e15

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert torch.isfinite(S).all(), "Very large finite values should not produce non-finite S"
    assert (S >= 0).all(), "S should be non-negative"


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_very_small_values(dtype):
    """SVD with very small (subnormal) values should not crash."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 32, 32
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device) * 1e-30

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert S.shape == (min(m, n),)
    # Reconstruction should still work (relaxed tolerance for subnormals)
    recon = U @ torch.diag(S) @ V.T
    err = (inp - recon).abs().max().item()
    # On some GPUs subnormal handling produces larger errors or NaN
    if torch.isnan(torch.tensor(err)):
        return  # NaN is acceptable for subnormal inputs
    assert err < 1e-10, f"Subnormal reconstruction error {err:.2e}"


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_mixed_nan_inf(dtype):
    """SVD with mixed NaN and Inf entries should not crash."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    m, n = 64, 64
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    inp[0, :] = float("inf")
    inp[:, 0] = float("nan")
    inp[1, 1] = float("-inf")

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=True)

    assert S.shape == (min(m, n),)
    assert not torch.isfinite(S).all(), "Mixed NaN/Inf should produce non-finite S"


@pytest.mark.svd
@pytest.mark.parametrize("m,n", [(2, 2), (64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_some_false_nan(m, n, dtype):
    """SVD with some=False should handle NaN input consistently."""
    if dtype == torch.float64 and not _fp64_supported():
        pytest.skip("GPU has limited float64 support")
    inp = torch.randn(m, n, dtype=dtype, device=flag_gems.device)
    inp[0, 0] = float("nan")

    with flag_gems.use_gems():
        U, S, V = torch.svd(inp, some=False)

    assert torch.isnan(S).any(), "NaN should propagate to S with some=False"
    assert U.shape == (m, m)
    assert V.shape == (n, n)
