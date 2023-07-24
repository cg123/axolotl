import torch
import transformers

from transformers.models.llama.modeling_llama import LlamaModel


class LlamaComboScaledRope(torch.nn.Module):
    """
    references: https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test
                https://github.com/jquesnelle/scaled-rope
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scale=1,
        alpha=1,
        device=None,
    ):
        super().__init__()
        if alpha != 1:
            base = base * alpha ** (dim / (dim - 2))

        self.scale = 1 / scale
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        t *= self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

        self.offset = 0  # a special treat for later

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            t *= self.scale
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False
            )
        return (
            self.cos_cached[:, :, self.offset : self.offset + seq_len, ...].to(
                dtype=x.dtype
            ),
            self.sin_cached[:, :, self.offset : self.offset + seq_len, ...].to(
                dtype=x.dtype
            ),
        )


def llama_scale_rope(model: transformers.LlamaForCausalLM, **kwargs):
    kwargs.update({"device": model.device})
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = LlamaComboScaledRope(
            layer.self_attn.head_dim, **kwargs
        )


def llama_set_rope_offset(model: transformers.LlamaForCausalLM, offset: int):
    while hasattr(model, 'model') and not isinstance(model, LlamaModel):
        model = model.model
    
    for layer in model.layers:
        layer.self_attn.rotary_emb.offset = offset
