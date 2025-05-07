import types

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.activations import ACT2FN


LOG = get_logger("axolotl.monkeypatch.qwen3moe_scattermoe")


def patch_scattermoe(use_torch_compile: bool = True):
    try:
        from scattermoe.kernels import ops as sm_ops
        from scattermoe.parallel_experts import parallel_linear
    except ImportError:
        LOG.warning(
            "scattermoe not installed, skipping patching of Qwen3MoeSparseMoeBlock"
        )
        LOG.warning(
            "Please install scattermoe with `pip install scattermoe@git+"
            "https://github.com/shawntan/scattermoe.git@63b76a2f5f28c052fb4cd7c34479a54158354052` "
            "to enable this feature."
        )
        return

    def _scattermoe_forward(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(
            -1, hidden_dim
        )  # (num_tokens, hidden_dim)

        # get routing weights
        router_logits = self.gate(hidden_states_flat)  # (num_tokens, num_experts)
        routing_weights_softmax = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_routing_weights, selected_experts = torch.topk(
            routing_weights_softmax, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            top_k_routing_weights /= top_k_routing_weights.sum(dim=-1, keepdim=True)

        top_k_routing_weights = top_k_routing_weights.to(hidden_states.dtype)

        with torch.no_grad():
            # sorted_expert_idxs: (num_tokens * top_k) -> flat list of expert indices, sorted for locality
            # sorted_scattered_idxs: (num_tokens * top_k) -> indices to gather from original x to match sorted_expert_idxs
            sorted_expert_idxs, sorted_scattered_idxs = sm_ops.flatten_and_sort(
                selected_experts
            )
            padded_block_idxs, expert_offsets = sm_ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        # up/gate projection
        stacked_gate_weights = torch.stack(
            [expert.gate_proj.weight for expert in self.experts]
        )
        stacked_up_weights = torch.stack(
            [expert.up_proj.weight for expert in self.experts]
        )
        w1_t = torch.cat([stacked_gate_weights, stacked_up_weights], dim=1)
        w1_experts_combined = w1_t.transpose(
            1, 2
        )  # (num_experts, hidden_dim, 2 * moe_intermediate_size)
        hidden_act_and_up_grouped = parallel_linear(
            inputs=hidden_states_flat,
            expert_weights=w1_experts_combined,
            k=self.top_k,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            expert_offsets=expert_offsets,
            gates=None,
            grouped_in=False,
            grouped_out=True,
        )  # (num_tokens * top_k, 2 * moe_intermediate_size)

        # apply activation function
        gated_values, up_values = hidden_act_and_up_grouped.chunk(2, dim=-1)
        intermediate_grouped = self.experts[0].act_fn(gated_values) * up_values
        # intermediate_grouped: (num_tokens * top_k, moe_intermediate_size)

        # output projection
        all_down_weights = torch.stack(
            [expert.down_proj.weight for expert in self.experts]
        )
        w2_experts_combined = all_down_weights.transpose(
            1, 2
        )  # (num_experts, moe_intermediate_size, hidden_dim)
        final_expert_output_flat = parallel_linear(
            inputs=intermediate_grouped,
            expert_weights=w2_experts_combined,
            k=1,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            expert_offsets=expert_offsets,
            gates=top_k_routing_weights,
            grouped_in=True,
            grouped_out=False,
        )  # (num_tokens, hidden_dim)

        final_hidden_states = final_expert_output_flat.view(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    if use_torch_compile and hasattr(torch, "compile"):
        LOG.info(
            "Compiling Qwen3MoeSparseMoeBlock._scattermoe_forward with torch.compile"
        )
        try:
            _scattermoe_forward = torch.compile(
                _scattermoe_forward,
                mode="reduce-overhead",
            )
            LOG.info("Successfully compiled _scattermoe_forward.")
        except Exception as e:
            LOG.warning(
                "Failed to compile _scattermoe_forward with torch.compile. Using "
                "uncompiled version instead.",
                exc_info=e,
            )

    Qwen3MoeSparseMoeBlock.forward = _scattermoe_forward
    LOG.info("Applied scattermoe patch to Qwen3MoeSparseMoeBlock.forward.")
