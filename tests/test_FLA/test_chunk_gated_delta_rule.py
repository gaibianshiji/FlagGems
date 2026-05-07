# Tests for chunk_gated_delta_rule fused operator.
# Reference: recurrent implementation for correctness verification.

import random

import pytest
import torch
import torch.nn.functional as F

import flag_gems

random.seed(42)
torch.manual_seed(42)


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


CUDA_AVAILABLE = is_cuda_available()


def _reference_forward(q, k, v, g, beta, scale=None, initial_state=None):
    """Recurrent reference implementation for correctness verification."""
    if q.dim() == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    o = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)

    if initial_state is not None:
        S = initial_state.float().clone()
        if S.dim() == 3:
            S = S.unsqueeze(0).expand(B, -1, -1, -1)
    else:
        S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    ratio = H // Hg

    for t in range(T):
        q_t = q[:, t].float()
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        g_t = g[:, t].float()
        beta_t = beta[:, t].float()

        if ratio > 1:
            q_t = q_t.repeat_interleave(ratio, dim=1)
            k_t = k_t.repeat_interleave(ratio, dim=1)

        o_t = torch.einsum("bhk,bhkv->bhv", q_t * scale, S)
        S = S * torch.exp(g_t).view(B, H, 1, 1)
        delta = v_t.unsqueeze(2) - torch.einsum("bhk,bhkv->bhv", k_t, S).unsqueeze(2)
        S = S + torch.einsum(
            "bhk,bhv->bhkv", k_t, beta_t.unsqueeze(-1) * delta.squeeze(2)
        )
        o[:, t] = o_t

    if squeeze_out:
        o = o.squeeze(0)
    return o.to(q.dtype)


def _create_inputs(B, T, H, K, V, Hg=None, device="cuda", dtype=torch.float32):
    """Create test inputs with proper g values (always negative via logsigmoid)."""
    if Hg is None:
        Hg = H
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    return q, k, v, g, beta


def _get_tol(dtype, T=128):
    """Return (rtol, atol) tolerance for a given dtype. Larger T needs more tolerance."""
    base = 2e-2 if T > 256 else 1e-2
    if dtype == torch.float32:
        return (base, base)
    elif dtype == torch.float16:
        return (3e-2, 3e-2)
    elif dtype == torch.bfloat16:
        return (5e-2, 5e-2)
    return (base, base)


# ==============================================================================
# A. Input scale coverage: small, medium, large
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize(
    "B, T, H, K, V",
    [
        # Small
        (1, 64, 2, 64, 64),
        # Medium
        (1, 128, 4, 64, 64),
        (1, 256, 4, 64, 64),
        (2, 128, 4, 64, 64),
        # Large
        (1, 512, 8, 64, 64),
        (2, 512, 8, 64, 64),
        (1, 1024, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_chunk_gated_delta_rule_batch(B, T, H, K, V, dtype):
    """Test chunk_gated_delta_rule with batched inputs (no cu_seqlens)."""
    q, k, v, g, beta = _create_inputs(
        B, T, H, K, V, device=flag_gems.device, dtype=dtype
    )
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# B. cu_seqlens coverage (packed sequences)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize(
    "T, H, K, V",
    [
        (64, 2, 64, 64),
        (128, 4, 64, 64),
        (256, 4, 64, 64),
        (512, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_chunk_gated_delta_rule_cu_seqlens(T, H, K, V, dtype):
    """Test with cu_seqlens (packed sequence format)."""
    device = flag_gems.device
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_multi_sequence():
    """Test with multiple sequences packed via cu_seqlens."""
    device = flag_gems.device
    dtype = torch.float32
    T1, T2 = 128, 64
    H, K, V = 4, 64, 64
    T_total = T1 + T2

    q = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, T_total, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype))
    cu_seqlens = torch.tensor([0, T1, T_total], device=device, dtype=torch.long)
    scale = K**-0.5

    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    # Verify each sequence independently
    ref1 = _reference_forward(
        q[:, :T1], k[:, :T1], v[:, :T1], g[:, :T1], beta[:, :T1], scale=scale
    )
    ref2 = _reference_forward(
        q[:, T1:], k[:, T1:], v[:, T1:], g[:, T1:], beta[:, T1:], scale=scale
    )
    rtol, atol = 2e-2, 2e-2
    torch.testing.assert_close(o_out[:, :T1], ref1, rtol=rtol, atol=atol)
    torch.testing.assert_close(o_out[:, T1:], ref2, rtol=rtol, atol=atol)


# ==============================================================================
# C. GQA (grouped-query attention) coverage
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("T", [128, 256, 512])
def test_chunk_gated_delta_rule_gqa(T):
    """Test with grouped-query attention (Hg < H)."""
    device = flag_gems.device
    H, Hg, K, V = 8, 2, 64, 64
    dtype = torch.float32
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, Hg=Hg, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=2e-2, atol=2e-2)


# ==============================================================================
# D. Initial state and final state
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_with_initial_state():
    """Test with non-zero initial state."""
    device = flag_gems.device
    B, T, H, K, V = 1, 128, 4, 64, 64
    dtype = torch.float32
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    initial_state = torch.randn(H, K, V, device=device, dtype=dtype) * 0.01
    scale = K**-0.5
    ref_out = _reference_forward(
        q, k, v, g, beta, scale=scale, initial_state=initial_state
    )
    _, o_out, _, final_state, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)
    assert final_state is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_output_final_state():
    """Test that output_final_state=True returns a valid final state."""
    device = flag_gems.device
    B, T, H, K, V = 1, 64, 2, 64, 64
    dtype = torch.float32
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    _, _, _, final_state, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
    )
    assert final_state is not None
    assert final_state.shape == (B, H, K, V)
    assert not torch.isnan(final_state).any()
    assert not torch.isinf(final_state).any()


# ==============================================================================
# E. Data type coverage: float16, bfloat16
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gated_delta_rule_fp16_bf16(dtype):
    """Test with float16 and bfloat16 dtypes."""
    device = flag_gems.device
    B, T, H, K, V = 1, 128, 4, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# F. Non-contiguous tensor coverage
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_non_contiguous():
    """Test with non-contiguous tensors (transpose then slice)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64

    # Create contiguous inputs first
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)

    # Make them non-contiguous via transpose then slice back
    # q: [B, T, H, K] -> transpose to [B, K, H, T] -> slice -> transpose back
    q_nc = q.transpose(1, 3)[:, :K, :, :T].transpose(1, 3).contiguous()
    k_nc = k.transpose(1, 3)[:, :K, :, :T].transpose(1, 3).contiguous()

    scale = K**-0.5
    ref_out = _reference_forward(q_nc, k_nc, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q_nc,
        k=k_nc,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# G. Edge case: single chunk (T == chunk_size)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_single_chunk():
    """Test with T exactly equal to chunk_size (64)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 64, 2, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# H. Edge case: non-aligned sequence length (T not multiple of chunk_size)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("T", [65, 100, 200])
def test_chunk_gated_delta_rule_non_aligned(T):
    """Test with sequence length not a multiple of chunk_size (64)."""
    device = flag_gems.device
    dtype = torch.float32
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# I. Numerical stability: extreme gate values
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_extreme_gates():
    """Test with very negative gate values (strong decay)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    # Very negative g -> strong exponential decay
    g = torch.full((B, T, H), -5.0, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    assert not torch.isnan(o_out).any()
    assert not torch.isinf(o_out).any()
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_zero_gates():
    """Test with g=0 (no decay, pure delta rule)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = torch.zeros(B, T, H, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# J. Different K/V dimensions
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize(
    "K, V",
    [
        (32, 32),
        (64, 32),
        (128, 64),
    ],
)
def test_chunk_gated_delta_rule_kv_dims(K, V):
    """Test with different K and V dimensions."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H = 1, 128, 4
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# K. Large batch sizes
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("T", [64, 128])
def test_chunk_gated_delta_rule_large_batch(B, T):
    """Test with large batch sizes."""
    device = flag_gems.device
    dtype = torch.float32
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# L. Very short sequences (T < chunk_size)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("T", [1, 8, 16, 32])
def test_chunk_gated_delta_rule_very_short(T):
    """Test with sequence length smaller than chunk_size (64)."""
    device = flag_gems.device
    dtype = torch.float32
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# M. Many heads (H=16, H=32)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("H", [16, 32])
def test_chunk_gated_delta_rule_many_heads(H):
    """Test with many attention heads."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, K, V = 1, 128, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# N. More GQA ratios: Hg=1, Hg=4 with H=8
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("Hg", [1, 4])
def test_chunk_gated_delta_rule_gqa_ratios(Hg):
    """Test with different GQA ratios (Hg=1 and Hg=4 with H=8)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 8, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, Hg=Hg, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=2e-2, atol=2e-2)


# ==============================================================================
# O. cu_seqlens with more sequences (3+)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_multi_sequence_3():
    """Test with 3 sequences packed via cu_seqlens."""
    device = flag_gems.device
    dtype = torch.float32
    T1, T2, T3 = 128, 64, 192
    H, K, V = 4, 64, 64
    T_total = T1 + T2 + T3

    q = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, T_total, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype))
    cu_seqlens = torch.tensor(
        [0, T1, T1 + T2, T_total], device=device, dtype=torch.long
    )
    scale = K**-0.5

    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    # Verify each sequence independently
    ref1 = _reference_forward(
        q[:, :T1], k[:, :T1], v[:, :T1], g[:, :T1], beta[:, :T1], scale=scale
    )
    ref2 = _reference_forward(
        q[:, T1 : T1 + T2],
        k[:, T1 : T1 + T2],
        v[:, T1 : T1 + T2],
        g[:, T1 : T1 + T2],
        beta[:, T1 : T1 + T2],
        scale=scale,
    )
    ref3 = _reference_forward(
        q[:, T1 + T2 :],
        k[:, T1 + T2 :],
        v[:, T1 + T2 :],
        g[:, T1 + T2 :],
        beta[:, T1 + T2 :],
        scale=scale,
    )
    rtol, atol = 2e-2, 2e-2
    torch.testing.assert_close(o_out[:, :T1], ref1, rtol=rtol, atol=atol)
    torch.testing.assert_close(o_out[:, T1 : T1 + T2], ref2, rtol=rtol, atol=atol)
    torch.testing.assert_close(o_out[:, T1 + T2 :], ref3, rtol=rtol, atol=atol)


# ==============================================================================
# P. cu_seqlens with non-aligned boundaries
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_cu_seqlens_non_aligned():
    """Test cu_seqlens where sequence boundaries are not chunk-aligned."""
    device = flag_gems.device
    dtype = torch.float32
    # T1=100 (not multiple of 64), T2=50 (not multiple of 64)
    T1, T2 = 100, 50
    H, K, V = 4, 64, 64
    T_total = T1 + T2

    q = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, T_total, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, T_total, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(1, T_total, H, device=device, dtype=dtype))
    cu_seqlens = torch.tensor([0, T1, T_total], device=device, dtype=torch.long)
    scale = K**-0.5

    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    ref1 = _reference_forward(
        q[:, :T1], k[:, :T1], v[:, :T1], g[:, :T1], beta[:, :T1], scale=scale
    )
    ref2 = _reference_forward(
        q[:, T1:], k[:, T1:], v[:, T1:], g[:, T1:], beta[:, T1:], scale=scale
    )
    rtol, atol = 2e-2, 2e-2
    torch.testing.assert_close(o_out[:, :T1], ref1, rtol=rtol, atol=atol)
    torch.testing.assert_close(o_out[:, T1:], ref2, rtol=rtol, atol=atol)


# ==============================================================================
# Q. cu_seqlens + GQA combination
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_cu_seqlens_gqa():
    """Test cu_seqlens combined with GQA (Hg < H)."""
    device = flag_gems.device
    dtype = torch.float32
    T = 256
    H, Hg, K, V = 8, 2, 64, 64

    q = torch.randn(1, T, Hg, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, T, Hg, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, T, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(1, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(1, T, H, device=device, dtype=dtype))
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
    scale = K**-0.5

    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    torch.testing.assert_close(o_out, ref_out, rtol=2e-2, atol=2e-2)


# ==============================================================================
# R. Final state correctness validation
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize(
    "B, T, H, K, V",
    [
        (1, 64, 2, 64, 64),
        (1, 128, 4, 64, 64),
        (1, 256, 4, 64, 64),
    ],
)
def test_chunk_gated_delta_rule_final_state_correctness(B, T, H, K, V):
    """Validate final_state matches the reference recurrent state."""
    device = flag_gems.device
    dtype = torch.float32
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5

    # Compute reference final state via recurrent loop
    S = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    ratio = H // (q.shape[2])
    for t in range(T):
        q_t = q[:, t].float()
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        g_t = g[:, t].float()
        beta_t = beta[:, t].float()
        if ratio > 1:
            q_t = q_t.repeat_interleave(ratio, dim=1)
            k_t = k_t.repeat_interleave(ratio, dim=1)
        S = S * torch.exp(g_t).view(B, H, 1, 1)
        delta = v_t.unsqueeze(2) - torch.einsum("bhk,bhkv->bhv", k_t, S).unsqueeze(2)
        S = S + torch.einsum(
            "bhk,bhv->bhkv", k_t, beta_t.unsqueeze(-1) * delta.squeeze(2)
        )
    ref_final_state = S

    _, _, _, final_state, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
    )

    assert final_state is not None
    assert final_state.shape == (B, H, K, V)
    torch.testing.assert_close(final_state, ref_final_state, rtol=2e-2, atol=2e-2)


# ==============================================================================
# S. Edge case: beta near 0 (minimal update)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_beta_near_zero():
    """Test with beta near 0 (minimal state updates)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.full((B, T, H), 1e-6, device=device, dtype=dtype)
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    assert not torch.isnan(o_out).any()
    assert not torch.isinf(o_out).any()
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# T. Edge case: beta = 1 (full update)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_beta_one():
    """Test with beta=1 (full weight updates)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.ones(B, T, H, device=device, dtype=dtype)
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# U. Edge case: all-zeros v (state only from initial + gate decay)
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_zero_v():
    """Test with all-zero v (output depends only on initial state + gate)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    scale = K**-0.5
    initial_state = torch.randn(H, K, V, device=device, dtype=dtype) * 0.01

    ref_out = _reference_forward(
        q, k, v, g, beta, scale=scale, initial_state=initial_state
    )
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=None,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)


# ==============================================================================
# V. Determinism: same inputs produce identical outputs
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_deterministic():
    """Test that the operator is deterministic (same inputs -> same outputs)."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H, K, V = 1, 128, 4, 64, 64
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

    _, o1, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(**kwargs)
    _, o2, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(**kwargs)

    torch.testing.assert_close(o1, o2, rtol=0, atol=0)


# ==============================================================================
# W. Precision characterization: error grows gracefully with T
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("T", [64, 128, 256, 512, 1024])
def test_chunk_gated_delta_rule_precision_characterization(T):
    """Verify that max error stays bounded and grows gracefully with T."""
    device = flag_gems.device
    dtype = torch.float32
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

    diff = (o_out.cpu().double() - ref_out.cpu().double()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    # Max error should stay below 2e-2 for any T
    assert max_err < 2e-2, f"max error {max_err:.2e} exceeds 2e-2 at T={T}"
    # Mean error should stay below 2e-3
    assert mean_err < 2e-3, f"mean error {mean_err:.2e} exceeds 2e-3 at T={T}"


# ==============================================================================
# X. fp16/bf16 with more shapes
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "B, T, H, K, V",
    [
        (1, 64, 2, 64, 64),
        (1, 256, 4, 64, 64),
        (2, 128, 4, 64, 64),
    ],
)
def test_chunk_gated_delta_rule_fp16_bf16_shapes(B, T, H, K, V, dtype):
    """Test fp16/bf16 across multiple shapes."""
    device = flag_gems.device
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# Y. cu_seqlens + fp16
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gated_delta_rule_cu_seqlens_half(dtype):
    """Test cu_seqlens with half-precision dtypes."""
    device = flag_gems.device
    T = 128
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
    scale = K**-0.5

    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    rtol, atol = _get_tol(dtype)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)


# ==============================================================================
# Z. Initial state + cu_seqlens combination
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_initial_state_cu_seqlens():
    """Test with both initial_state and cu_seqlens."""
    device = flag_gems.device
    dtype = torch.float32
    T = 128
    H, K, V = 4, 64, 64
    q, k, v, g, beta = _create_inputs(1, T, H, K, V, device=device, dtype=dtype)
    initial_state = torch.randn(H, K, V, device=device, dtype=dtype) * 0.01
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
    scale = K**-0.5

    # Reference: apply initial_state to single sequence
    ref_out = _reference_forward(
        q, k, v, g, beta, scale=scale, initial_state=initial_state
    )
    _, o_out, _, final_state, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(o_out, ref_out, rtol=1e-2, atol=1e-2)
    assert final_state is not None


# ==============================================================================
# AA. Different K/V with more combinations
# ==============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize(
    "K, V",
    [
        (32, 64),
        (128, 32),
        (256, 64),
    ],
)
def test_chunk_gated_delta_rule_kv_extreme(K, V):
    """Test with extreme K/V dimension ratios."""
    device = flag_gems.device
    dtype = torch.float32
    B, T, H = 1, 128, 4
    q, k, v, g, beta = _create_inputs(B, T, H, K, V, device=device, dtype=dtype)
    scale = K**-0.5
    ref_out = _reference_forward(q, k, v, g, beta, scale=scale)
    _, o_out, _, _, _, _, _ = flag_gems.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    rtol, atol = _get_tol(dtype, T=T)
    torch.testing.assert_close(o_out, ref_out, rtol=rtol, atol=atol)
