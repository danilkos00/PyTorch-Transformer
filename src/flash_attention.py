import torch
import triton
import triton.language as tl
from einops import einsum


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    N_HEADS, N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # b_q
    K_TILE_SIZE: tl.constexpr, # b_k
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)

    batch_index = batch_head_index // N_HEADS
    head_index = batch_head_index % N_HEADS

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')

    for j in tl.range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        S_i = tl.dot(Q_i, tl.trans(K_j), input_precision='ieee') * scale

        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE
            k_idx = tl.arange(0, K_TILE_SIZE)[None, :] + j * K_TILE_SIZE
            mask = (q_idx >= k_idx) & (k_idx < N_KEYS)
            S_i = tl.where(mask, S_i, float('-inf'))

        m_prev = m_i + 0

        m_i = tl.maximum(m_i, tl.max(S_i, axis=-1))

        P_hat = tl.exp(S_i - m_i[:, None])

        coeff = tl.exp(m_prev - m_i)
        l_i = coeff * l_i + tl.sum(P_hat, axis=-1)

        O_i = O_i * coeff[:, None] + tl.dot(P_hat, V_j, input_precision='ieee')

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_dob, stride_doh, stride_doq, stride_dod,
    stride_lb, stride_lh, stride_lq,
    N_HEADS, N_QUERIES, N_KEYS,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # B_q
    K_TILE_SIZE: tl.constexpr, # B_k
    is_causal: tl.constexpr
):
    key_tile_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb + head_idx * stride_qh,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb + head_idx * stride_kh,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb + head_idx * stride_vh,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob + head_idx * stride_oh,
        shape=(N_QUERIES, DIM),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb + head_idx * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    base_dQ_ptr = dQ_ptr + batch_idx * stride_qb + head_idx * stride_qh
    dQ_row_idx = tl.arange(0, Q_TILE_SIZE)
    dQ_col_idx = tl.arange(0, DIM)

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx * stride_kb + head_idx * stride_kh,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx * stride_vb + head_idx * stride_vh,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob + head_idx * stride_doh,
        shape=(N_QUERIES, DIM),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

    dK = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)

    for i in tl.range(0, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option='zero')
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')

        D = tl.sum(dO * O, axis=-1)

        attn = tl.dot(Q, tl.trans(K), input_precision='ieee') * scale

        attn_probs = tl.exp(attn - L[:, None])

        if is_causal:
            q_idx = tl.arange(0, Q_TILE_SIZE)[:, None] + i * Q_TILE_SIZE
            k_idx = tl.arange(0, K_TILE_SIZE)[None, :] + key_tile_idx * K_TILE_SIZE
            mask = (q_idx >= k_idx) & (k_idx < N_KEYS)
            attn_probs = tl.where(mask, attn_probs, 0.0)

        dV = tl.dot(tl.trans(attn_probs), dO, acc=dV, input_precision='ieee')

        d_attn_probs = tl.dot(dO, tl.trans(V), input_precision='ieee')
        d_attn = attn_probs * (d_attn_probs - D[:, None]) * scale

        dQ_offsets = (
            (i * Q_TILE_SIZE + dQ_row_idx[:, None]) * stride_qq +
            dQ_col_idx[None, :] * stride_qd
        )

        tl.atomic_add(
            base_dQ_ptr + dQ_offsets,
            tl.dot(d_attn, K, input_precision='ieee')
        )

        dK = tl.dot(tl.trans(d_attn), Q, acc=dK, input_precision='ieee')

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV, boundary_check=(0, 1))


class TritonFlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda, 'Tensors must be on the cuda device'
        B, heads, q_seq_len, DIM = Q.size()
        k_seq_len = K.size(-2)
        num_warps = 4 if DIM <= 64 else 8
        SCALE = 1 / DIM**0.5

        B_Q = 64
        B_K = 32
        ctx.is_causal = is_causal

        O = torch.empty_like(Q)
        L = torch.empty((B, heads, q_seq_len), device=Q.device, dtype=Q.dtype)

        grid = lambda meta: (triton.cdiv(meta['N_QUERIES'], meta['Q_TILE_SIZE']), B * meta['N_HEADS'])

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads, q_seq_len, k_seq_len,
            SCALE,
            DIM,
            B_Q, B_K,
            ctx.is_causal,
            num_warps=num_warps,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, heads, q_seq_len, DIM = Q.size()
        k_seq_len = K.size(-2)

        num_warps = 4

        SCALE = 1 / DIM**0.5

        B_Q = 32
        B_K = 32

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        grid = lambda meta: (triton.cdiv(meta['N_KEYS'], meta['K_TILE_SIZE']), B * meta['N_HEADS'])

        flash_bwd_kernel[grid](
            Q, K, V,
            O, dO, L,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads, q_seq_len, k_seq_len,
            SCALE,
            DIM=DIM,
            Q_TILE_SIZE=B_Q,
            K_TILE_SIZE=B_K,
            is_causal=ctx.is_causal,
            num_warps=num_warps,
        )

        return dQ, dK, dV, None
    

class PytorchFlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda, 'Tensors must be on the cuda device'
        B, heads, q_seq_len, DIM = Q.size()
        k_seq_len = K.size(-2)
        num_warps = 4 if DIM <= 64 else 8
        SCALE = 1 / DIM**0.5

        B_Q = 64
        B_K = 32
        ctx.is_causal = is_causal

        O = torch.empty_like(Q)
        L = torch.empty((B, heads, q_seq_len), device=Q.device, dtype=Q.dtype)

        grid = lambda meta: (triton.cdiv(meta['N_QUERIES'], meta['Q_TILE_SIZE']), B * meta['N_HEADS'])

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            heads, q_seq_len, k_seq_len,
            SCALE,
            DIM,
            B_Q, B_K,
            ctx.is_causal,
            num_warps=num_warps,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        return flash_backward_pytorch(Q, K, V, O, dO, L, ctx.is_causal)
    

def flash_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    scale = 1 / Q.size(-1)**0.5

    D = O.mul(dO).sum(-1)

    S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale

    P = torch.exp(S - L[..., None])

    if is_causal:
        mask = torch.ones_like(S, dtype=torch.bool).tril()
        P = torch.where(mask, P, 0.0)

    dV = einsum(P, dO, '... q k, ... q d -> ... k d')

    dP = einsum(dO, V, '... k d, ... q d -> ... k q')

    dS = P.mul(dP - D[..., None])

    dQ = einsum(dS, K, '... q k, ... k d -> ... q d') * scale

    dK = einsum(dS, Q, '... q k, ... q d -> ... k d') * scale

    return dQ, dK, dV, None