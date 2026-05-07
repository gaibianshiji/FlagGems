# SPDX-License-Identifier: Apache-2.0
# PyTorch-native implementation of chunk_gated_delta_rule for GPUs where
# Triton's tl.dot + block_ptr produces incorrect results (e.g. Iluvatar BI-V150).
# ruff: noqa: E501

import torch
import torch.nn.functional as F


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Compute chunk-local cumulative sum of g."""
    if cu_seqlens is not None:
        output = torch.empty_like(g)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            g_seq = g[bos:eos]
            T = eos - bos
            NT = (T + chunk_size - 1) // chunk_size
            pad_len = NT * chunk_size - T
            if pad_len > 0:
                g_seq = F.pad(g_seq, (0, 0, 0, pad_len))
            g_seq = g_seq.reshape(NT, chunk_size, *g_seq.shape[1:])
            output[bos:eos] = g_seq.cumsum(dim=1).reshape(
                NT * chunk_size, *g_seq.shape[2:]
            )[:T]
        return output
    else:
        B, T, H = g.shape
        NT = (T + chunk_size - 1) // chunk_size
        pad_len = NT * chunk_size - T
        if pad_len > 0:
            g = F.pad(g, (0, 0, 0, pad_len))
        g = g.reshape(B, NT, chunk_size, H)
        g = g.cumsum(dim=2)
        g = g.reshape(B, NT * chunk_size, H)[:, :T]
        return g


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute chunk-scaled dot product KKT.

    A[i,j] = -(beta[i]*k[i]) @ k[j] * exp(g_cumsum[i] - g_cumsum[j])
    for i > j (strictly lower triangular), 0 otherwise.
    """
    if cu_seqlens is not None:
        T, Hg, K = k.shape
        H = beta.shape[-1]
        N = len(cu_seqlens) - 1
        BT = chunk_size
        A = torch.zeros(
            N, (T + BT - 1) // BT, BT, BT, H, device=k.device, dtype=output_dtype
        )

        for i_b in range(N):
            bos, eos = cu_seqlens[i_b].item(), cu_seqlens[i_b + 1].item()
            k_seq = k[bos:eos]
            beta_seq = beta[bos:eos]
            g_seq = g_cumsum[bos:eos]
            T_seq = eos - bos
            NT_seq = (T_seq + BT - 1) // BT

            for i_t in range(NT_seq):
                t_start = i_t * BT
                t_end = min(t_start + BT, T_seq)
                bt = t_end - t_start

                k_chunk = k_seq[t_start:t_end]
                beta_chunk = beta_seq[t_start:t_end]
                g_chunk = g_seq[t_start:t_end]

                ratio = H // Hg
                if ratio > 1:
                    k_expanded = k_chunk.repeat_interleave(ratio, dim=1)
                else:
                    k_expanded = k_chunk
                kb = k_expanded * beta_chunk.unsqueeze(-1)

                A_chunk = torch.bmm(
                    kb.float().permute(1, 0, 2),
                    k_expanded.float().permute(1, 2, 0),
                )
                A_chunk = A_chunk.permute(1, 2, 0)

                g_diff = g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0)
                A_chunk = A_chunk * torch.exp(g_diff.float().clamp(max=80.0))
                mask = torch.tril(
                    torch.ones(bt, bt, device=k.device, dtype=torch.bool), diagonal=-1
                )
                A_chunk = -A_chunk * mask.unsqueeze(-1)
                A[i_b, i_t, :bt, :bt] = A_chunk
        return A

    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    pad_len = NT * BT - T
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))
        g_cumsum = F.pad(g_cumsum, (0, 0, 0, pad_len))

    ratio = H // Hg
    k_chunks = k.reshape(B * NT, BT, Hg, K)
    beta_chunks = beta.reshape(B * NT, BT, H)
    g_chunks = g_cumsum.reshape(B * NT, BT, H)

    if ratio > 1:
        k_expanded = k_chunks.repeat_interleave(ratio, dim=2)
    else:
        k_expanded = k_chunks

    kb = k_expanded.float() * beta_chunks.unsqueeze(-1).float()

    A = torch.bmm(
        kb.permute(0, 2, 1, 3).reshape(B * NT * H, BT, K),
        k_expanded.float().permute(0, 2, 3, 1).reshape(B * NT * H, K, BT),
    )

    g_diff = g_chunks.unsqueeze(2) - g_chunks.unsqueeze(1)
    A = A.reshape(B * NT, H, BT, BT).permute(0, 2, 3, 1)
    A = A * torch.exp(g_diff.float().clamp(max=80.0))

    mask = torch.tril(
        torch.ones(BT, BT, device=k.device, dtype=torch.bool), diagonal=-1
    )
    A = -A * mask.unsqueeze(0).unsqueeze(-1)

    A = A.reshape(B, NT, BT, BT, H)
    return A.to(output_dtype)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute (I - A)^{-1} for lower triangular A. Fully batched."""
    B, NT, BT, _, H = A.shape
    eye = torch.eye(BT, device=A.device, dtype=output_dtype)

    L = (
        A.reshape(B * NT, BT, BT, H)
        .permute(0, 3, 1, 2)
        .reshape(B * NT * H, BT, BT)
        .float()
    )
    IL = eye.unsqueeze(0) - L
    eye_batch = eye.unsqueeze(0).expand(B * NT * H, -1, -1)
    try:
        result = torch.linalg.solve_triangular(IL, eye_batch, upper=False)
    except Exception:
        result = torch.linalg.inv(IL)
    A = result.reshape(B * NT, H, BT, BT).permute(0, 2, 3, 1).reshape(B, NT, BT, BT, H)
    return A.to(output_dtype)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute w and u for WY representation. Fully batched."""
    if cu_seqlens is not None:
        T, Hg, K = k.shape
        H = v.shape[-2]
        V = v.shape[-1]
        BT = chunk_size
        N = len(cu_seqlens) - 1

        w = torch.zeros(T, H, K, device=k.device, dtype=torch.float32)
        u = torch.zeros(T, H, V, device=v.device, dtype=torch.float32)

        for i_b in range(N):
            bos, eos = cu_seqlens[i_b].item(), cu_seqlens[i_b + 1].item()
            k_seq = k[bos:eos]
            v_seq = v[bos:eos]
            beta_seq = beta[bos:eos]
            T_seq = eos - bos
            NT_seq = (T_seq + BT - 1) // BT

            for i_t in range(NT_seq):
                t_start = i_t * BT
                t_end = min(t_start + BT, T_seq)
                bt = t_end - t_start

                k_chunk = k_seq[t_start:t_end]
                v_chunk = v_seq[t_start:t_end]
                beta_chunk = beta_seq[t_start:t_end]
                A_chunk = A[i_b, i_t, :bt, :bt]

                ratio = H // Hg
                if ratio > 1:
                    k_expanded = k_chunk.repeat_interleave(ratio, dim=1)
                else:
                    k_expanded = k_chunk

                w_chunk = beta_chunk.unsqueeze(-1) * k_expanded
                A_t = A_chunk.permute(2, 0, 1).float()
                w_t = w_chunk.permute(1, 0, 2).float()
                w_chunk = (A_t @ w_t).permute(1, 0, 2)

                u_chunk = beta_chunk.unsqueeze(-1) * v_chunk
                u_t = u_chunk.permute(1, 0, 2).float()
                u_chunk = (A_t @ u_t).permute(1, 0, 2)

                w[bos + t_start : bos + t_end] = w_chunk
                u[bos + t_start : bos + t_end] = u_chunk

        return w.to(k.dtype), u.to(v.dtype)

    B, T, Hg, K = k.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    pad_len = NT * BT - T
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    ratio = H // Hg
    k_chunks = k.reshape(B * NT, BT, Hg, K)
    v_chunks = v.reshape(B * NT, BT, H, V)
    beta_chunks = beta.reshape(B * NT, BT, H)

    if ratio > 1:
        k_expanded = k_chunks.repeat_interleave(ratio, dim=2)
    else:
        k_expanded = k_chunks

    w_chunks = beta_chunks.unsqueeze(-1) * k_expanded
    A_t = (
        A.reshape(B * NT, BT, BT, H)
        .permute(0, 3, 1, 2)
        .reshape(B * NT * H, BT, BT)
        .float()
    )
    w_t = w_chunks.permute(0, 2, 1, 3).reshape(B * NT * H, BT, K).float()
    w_t = torch.bmm(A_t, w_t)
    w = (
        w_t.reshape(B * NT, H, BT, K)
        .permute(0, 2, 1, 3)
        .reshape(B, NT * BT, H, K)[:, :T]
    )

    u_chunks = beta_chunks.unsqueeze(-1) * v_chunks
    u_t = u_chunks.permute(0, 2, 1, 3).reshape(B * NT * H, BT, V).float()
    u_t = torch.bmm(A_t, u_t)
    u = (
        u_t.reshape(B * NT, H, BT, V)
        .permute(0, 2, 1, 3)
        .reshape(B, NT * BT, H, V)[:, :T]
    )

    return w.to(k.dtype), u.to(v.dtype)


def chunk_gated_delta_rule_fwd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute output and state update using chunked algorithm.

    For each chunk:
      1. Snapshot inter-chunk state h
      2. Compute v_new[t] = u[t] - w[t] * exp(g_cumsum[t]) @ h  (delta)
      3. Compute o_inter = (q * scale * exp(g_cumsum)) @ h        (inter-chunk)
      4. Compute S_attn[i,j] = (q[i]*scale @ k[j]) * exp(g[i]-g[j])  (intra-chunk)
      5. Compute o_intra = S_attn @ v_new                          (intra-chunk)
      6. Output: o = o_inter + o_intra
      7. Update: S = h * exp(g_last) + k_scaled^T @ v_new
    """
    if cu_seqlens is not None:
        if q.dim() == 4 and q.shape[0] == 1:
            q = q.squeeze(0)
        if k.dim() == 4 and k.shape[0] == 1:
            k = k.squeeze(0)
        if w.dim() == 4 and w.shape[0] == 1:
            w = w.squeeze(0)
        if u.dim() == 4 and u.shape[0] == 1:
            u = u.squeeze(0)

    if cu_seqlens is not None:
        T, Hg, K = q.shape
        B = 1
    else:
        B, T, Hg, K = q.shape
    H = u.shape[-2]
    V = u.shape[-1]
    BT = chunk_size

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
    else:
        N = B

    if scale is None:
        scale = K**-0.5

    h = torch.zeros(
        N, (T + BT - 1) // BT, H, K, V, device=k.device, dtype=torch.float32
    )
    final_state = None
    if cu_seqlens is not None:
        o = torch.zeros(T, H, V, device=k.device, dtype=k.dtype)
    else:
        o = torch.zeros(B, T, H, V, device=k.device, dtype=k.dtype)

    if initial_state is not None:
        h_state_all = initial_state.float().clone()
        if h_state_all.dim() == 3:
            h_state_all = h_state_all.unsqueeze(0).expand(N, -1, -1, -1).clone()
    else:
        h_state_all = torch.zeros(N, H, K, V, device=k.device, dtype=torch.float32)

    full_mask = torch.tril(torch.ones(BT, BT, device=k.device, dtype=torch.bool))
    ratio = H // Hg
    need_expand = ratio > 1

    with torch.no_grad():
        for i_n in range(N):
            h_state = h_state_all[i_n]

            if cu_seqlens is not None:
                bos, eos = cu_seqlens[i_n].item(), cu_seqlens[i_n + 1].item()
                q_seq = q[bos:eos]
                k_seq = k[bos:eos]
                w_seq = w[bos:eos]
                u_seq = u[bos:eos]
                g_seq = g_cumsum[bos:eos]
                T_seq = eos - bos
            else:
                q_seq = q[i_n]
                k_seq = k[i_n]
                w_seq = w[i_n]
                u_seq = u[i_n]
                g_seq = g_cumsum[i_n]
                T_seq = T

            NT_seq = (T_seq + BT - 1) // BT
            h_state_f = h_state.float()

            for i_t in range(NT_seq):
                t_start = i_t * BT
                t_end = min(t_start + BT, T_seq)
                bt = t_end - t_start

                q_chunk = q_seq[t_start:t_end]
                k_chunk = k_seq[t_start:t_end]
                w_chunk = w_seq[t_start:t_end]
                u_chunk = u_seq[t_start:t_end]
                g_chunk = g_seq[t_start:t_end]

                if need_expand:
                    q_expanded = q_chunk.repeat_interleave(ratio, dim=1)
                    k_expanded = k_chunk.repeat_interleave(ratio, dim=1)
                else:
                    q_expanded = q_chunk
                    k_expanded = k_chunk

                h[i_n, i_t] = h_state

                g_last = g_chunk[-1]
                g_exp = torch.exp(g_chunk.float().clamp(max=80.0))
                q_scaled = q_expanded.float() * scale

                w_scaled_t = (w_chunk.float() * g_exp.unsqueeze(-1)).transpose(0, 1)
                qe_t = (q_scaled * g_exp.unsqueeze(-1)).transpose(0, 1)

                wh_t = torch.matmul(w_scaled_t, h_state_f)
                o_inter_t = torch.matmul(qe_t, h_state_f)

                v_new_t = u_chunk.float().transpose(0, 1) - wh_t

                k_expanded_t = k_expanded.float().transpose(0, 1)
                S_attn_t = torch.bmm(
                    q_scaled.transpose(0, 1), k_expanded_t.transpose(1, 2)
                )
                g_diff = g_chunk.unsqueeze(1).float() - g_chunk.unsqueeze(0).float()
                S_attn_t = S_attn_t * torch.exp(g_diff.clamp(max=80.0)).permute(2, 0, 1)
                mask = full_mask[:bt, :bt].unsqueeze(0)
                S_attn_t = S_attn_t * mask

                o_intra_t = torch.bmm(S_attn_t, v_new_t)

                o_chunk = (o_inter_t + o_intra_t).transpose(0, 1).to(o.dtype)

                if cu_seqlens is not None:
                    o[bos + t_start : bos + t_end] = o_chunk
                else:
                    o[i_n, t_start:t_end] = o_chunk

                g_last_exp = torch.exp(g_last.float().clamp(max=80.0))
                g_diff_last = torch.exp((g_last - g_chunk).float().clamp(max=80.0))
                k_scaled_t = k_expanded_t * g_diff_last.transpose(0, 1).unsqueeze(-1)
                h_cumsum = torch.bmm(k_scaled_t.transpose(1, 2), v_new_t)
                h_state_f = h_state_f * g_last_exp.view(H, 1, 1) + h_cumsum
                h_state = h_state_f

            h_state_all[i_n] = h_state

    if output_final_state:
        final_state = h_state_all.to(k.dtype)

    return h.to(k.dtype), o, final_state


def chunk_gated_delta_rule_native_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """PyTorch-native chunk_gated_delta_rule forward pass.

    Fallback for GPUs where Triton tl.dot + block_ptr produces incorrect
    results (e.g. Iluvatar BI-V150). Uses batched matmul instead of Triton
    kernels for all compute-intensive steps.
    """
    chunk_size = 64

    squeezed = False
    if cu_seqlens is not None:
        if q.dim() == 4 and q.shape[0] == 1:
            q = q.squeeze(0)
            squeezed = True
        if k.dim() == 4 and k.shape[0] == 1:
            k = k.squeeze(0)
        if v.dim() == 4 and v.shape[0] == 1:
            v = v.squeeze(0)
        if g.dim() == 3 and g.shape[0] == 1:
            g = g.squeeze(0)
        if beta.dim() == 3 and beta.shape[0] == 1:
            beta = beta.squeeze(0)

    g_cumsum = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=torch.float32)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    h, o, final_state = chunk_gated_delta_rule_fwd_fused(
        q=q,
        k=k,
        w=w,
        u=u,
        g_cumsum=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    if squeezed:
        o = o.unsqueeze(0)
        g_cumsum = g_cumsum.unsqueeze(0)

    return g_cumsum, o, A, final_state, None, None, None
