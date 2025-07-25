# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
    elif major>=9:
        from sageattention_sm90 import sageattn
    @torch.compiler.disable()
    def sageattn_func(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        if q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
except Exception as e:
    try:
        from sageattention import sageattn
        def sageattn_func(q, k, v, attn_mask=None, dropout_p=0, is_causal=False):
            if q.dtype == torch.float32:
                return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
            else:
                return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    except:
        print(f"Warning: Could not load sageattention: {str(e)}")
        if isinstance(e, ModuleNotFoundError):
            print("sageattention package is not installed")
        elif isinstance(e, ImportError) and "DLL" in str(e):
            print("sageattention DLL loading error")
        sageattn_func = None
import warnings

__all__ = [
    'flash_attention',
    'attention',
]

from einops import rearrange
from flash_attn import flash_attn_func

def natten_attention(query, key, value, grid_sizes):
    import natten
    num_frames, height, width = grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2]
    mode = "bs (f h w) hn hd -> bs f h w hn hd"
    query = rearrange(query, mode, f=num_frames, h=height, w=width)
    key = rearrange(key, mode, f=num_frames, h=height, w=width)
    value = rearrange(value, mode, f=num_frames, h=height, w=width)
    attn = natten.na3d(query, key, value, backend="cutlass-fna", kernel_size=(18, 24, 24), stride=(16, 8, 8))
    attn = rearrange(attn, "bs f h w hn hd -> bs (f h w) hn hd", f=num_frames, h=height, w=width)
    return attn

def swa_flash_attention(query, key, value, grid_sizes, windows_size=4096):
    out_dtype = query.dtype
    num_frames, height, width = grid_sizes[0][0], grid_sizes[0][1], grid_sizes[0][2]

    querys = torch.tensor_split(query.to(torch.bfloat16), 6, 2)
    keys = torch.tensor_split(key.to(torch.bfloat16), 6, 2)
    values = torch.tensor_split(value.to(torch.bfloat16), 6, 2)
    new_querys = []
    new_keys = []
    new_values = []
    for index, mode in enumerate(
        [
            "bs (f h w) hn hd -> bs (h w f) hn hd", 
            "bs (f h w) hn hd -> bs (w h f) hn hd",
            "bs (f h w) hn hd -> bs (h f w) hn hd", 
            "bs (f h w) hn hd -> bs (w f h) hn hd",
            "bs (f h w) hn hd -> bs (f h w) hn hd", 
            "bs (f h w) hn hd -> bs (f w h) hn hd",
        ]
    ):
        
        new_querys.append(rearrange(querys[index], mode, f=num_frames, h=height, w=width))
        new_keys.append(rearrange(keys[index], mode, f=num_frames, h=height, w=width))
        new_values.append(rearrange(values[index], mode, f=num_frames, h=height, w=width))
    query = torch.cat(new_querys, dim=2)
    key = torch.cat(new_keys, dim=2)
    value = torch.cat(new_values, dim=2)
    hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False, window_size=(windows_size, windows_size))
    hidden_states = torch.tensor_split(hidden_states, 6, 2)
    new_hidden_states = []
    for index, mode in enumerate(
        [
            "bs (h w f) hn hd -> bs (f h w) hn hd", 
            "bs (w h f) hn hd -> bs (f h w) hn hd",
            "bs (h f w) hn hd -> bs (f h w) hn hd", 
            "bs (w f h) hn hd -> bs (f h w) hn hd",
            "bs (f h w) hn hd -> bs (f h w) hn hd", 
            "bs (f w h) hn hd -> bs (f h w) hn hd",
        ]
    ):
        new_hidden_states.append(rearrange(hidden_states[index], mode, f=num_frames, h=height, w=width))
    hidden_states = torch.cat(new_hidden_states, dim=2)
    return hidden_states.to(out_dtype)

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    #assert dtype in half_dtypes
    #assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    attention_mode='sdpa',
):  
    if "flash" in attention_mode:
        if attention_mode == 'flash_attn_2':
            fa_version = 2
        elif attention_mode == 'flash_attn_3':
            fa_version = 3
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    elif attention_mode == 'sdpa':
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()
    else:
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()
