from einops import rearrange
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
import torch
from torch import nn


def neox_flash_attn(
    self: GPTNeoXAttention, query, key, value, attention_mask=None, head_mask=None
):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    assert head_mask is None

    with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )

    return (attn_output, None)

    nn.functional.scaled_dot_product_attention()
    qkv = torch.stack([query, key, value], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    qkv = qkv.bfloat16()

    if not attention_mask:
        attn_scores = flash_attn_qkvpacked_func(
            qkv, softmax_scale=1 / self.norm_factor, causal=True
        )
    else:
        bsz, q_len, nheads, _ = qkv.shape

        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, attention_mask)
        x_unpad = rearrange(
            x_unpad,
            "nnz (three h d) -> nnz three h d",
            three=3,
            h=nheads,
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_q_lens,
            max_s,
            dropout_p=0.0,
            softmax_scale=1 / self.norm_factor,
            causal=True,
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                indices,
                bsz,
                q_len,
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )

    attn_scores = attn_scores.transpose(1, 2)  # [bsz, nh, q_len, hd]
    attn_weights = attn_weights.to(value.dtype)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = self.attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def patch_neox_flash_attn():
    GPTNeoXAttention._attn = neox_flash_attn
