import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.activations import ACT2FN

LOG = get_logger("axolotl.monkeypatch.qwen3moe_scattermoe_class")


def patch_qwen3moe_scattermoe():
    try:
        from scattermoe.mlp import GLUMLP
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

    class ScattermoeGLUBlock(nn.Module):
        def __init__(self, config: Qwen3MoeConfig):
            super().__init__()
            self.config = config
            self.top_k = config.num_experts_per_tok

            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.moe_mlp = GLUMLP(
                input_size=config.hidden_size,
                hidden_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=self.top_k,
                activation=ACT2FN[config.hidden_act],
            )

            self.register_load_state_dict_pre_hook(self._load_state_hook)
            self.register_state_dict_post_hook(self._state_hook)

        def forward(self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states_flat)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.config.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states_flat = self.moe_mlp(
                hidden_states_flat, routing_weights, selected_experts
            )
            final_hidden_states = final_hidden_states_flat.view(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states, router_logits

        @staticmethod
        def _load_state_hook(
            module: nn.Module,
            state_dict: dict,
            prefix: str,
            local_metadata: dict,
            strict: bool,
            missing_keys: set,
            unexpected_keys: set,
            error_msgs: list,
        ) -> None:
            LOG.warning(f"[{prefix}] Hook called. prefix: {prefix}, module: {module}")
            if not isinstance(module, ScattermoeGLUBlock):
                return
            LOG.info(f"[{prefix}] Hook started.")
            LOG.info(f"[{prefix}] Initial missing_keys: {missing_keys}")
            LOG.info(f"[{prefix}] Initial unexpected_keys: {unexpected_keys}")

            num_experts = module.config.num_experts

            scattermoe_w1_key = prefix + "moe_mlp.experts.weight"
            scattermoe_w2_key = prefix + "moe_mlp.output_experts.weight"

            gate_proj_weights = []
            up_proj_weights = []
            down_proj_weights = []

            keys_to_remove_from_state_dict = []

            found_any_hf_expert_weights = False
            for i in range(num_experts):
                hf_gate_key = prefix + f"experts.{i}.gate_proj.weight"
                hf_up_key = prefix + f"experts.{i}.up_proj.weight"
                hf_down_key = prefix + f"experts.{i}.down_proj.weight"

                if hf_gate_key in state_dict and hf_up_key in state_dict:
                    found_any_hf_expert_weights = True
                    gate_proj_weights.append(state_dict[hf_gate_key])
                    up_proj_weights.append(state_dict[hf_up_key])
                    keys_to_remove_from_state_dict.extend([hf_gate_key, hf_up_key])
                elif hf_gate_key in state_dict or hf_up_key in state_dict:
                    error_msgs.append(
                        f"Inconsistent gate/up projection weights for expert {i} in {prefix}"
                    )

                if hf_down_key in state_dict:
                    found_any_hf_expert_weights = True  # Can be true even if only down_proj is found, though less likely
                    down_proj_weights.append(state_dict[hf_down_key])
                    keys_to_remove_from_state_dict.append(hf_down_key)

            LOG.info(
                f"[{prefix}] found_any_hf_expert_weights: {found_any_hf_expert_weights}"
            )
            if not found_any_hf_expert_weights:
                # No HF expert weights found, maybe it's already in scattermoe format or a different checkpoint
                # If scattermoe keys are missing, load_state_dict will handle it.
                return

            LOG.info(f"[{prefix}] Num experts: {num_experts}")
            LOG.info(f"[{prefix}] len(gate_proj_weights): {len(gate_proj_weights)}")
            LOG.info(f"[{prefix}] len(up_proj_weights): {len(up_proj_weights)}")
            LOG.info(f"[{prefix}] len(down_proj_weights): {len(down_proj_weights)}")

            if gate_proj_weights and len(gate_proj_weights) != num_experts:
                error_msgs.append(
                    f"Expected {num_experts} gate_proj weights, but found {len(gate_proj_weights)} in {prefix}"
                )
            if up_proj_weights and len(up_proj_weights) != num_experts:
                error_msgs.append(
                    f"Expected {num_experts} up_proj weights, but found {len(up_proj_weights)} in {prefix}"
                )
            if down_proj_weights and len(down_proj_weights) != num_experts:
                error_msgs.append(
                    f"Expected {num_experts} down_proj weights, but found {len(down_proj_weights)} in {prefix}"
                )

            if (
                len(gate_proj_weights) == num_experts
                and len(up_proj_weights) == num_experts
            ):
                LOG.info(f"[{prefix}] Creating {scattermoe_w1_key}")
                # Combine gate_proj and up_proj for scattermoe's first ParallelExperts layer
                # Original Qwen3MoeMLP: gate_proj.weight (I_moe, H), up_proj.weight (I_moe, H)
                # scattermoe GLUMLP expects: experts.weight (E, 2*I_moe, H)
                combined_w1 = [
                    torch.cat([u, g], dim=0)
                    for g, u in zip(gate_proj_weights, up_proj_weights)
                ]
                state_dict[scattermoe_w1_key] = torch.stack(combined_w1, dim=0)
                if scattermoe_w1_key in missing_keys:
                    missing_keys.remove(scattermoe_w1_key)
                if (
                    scattermoe_w1_key in unexpected_keys
                ):  # Should not happen if loading HF checkpoint
                    LOG.warning(
                        f"Key {scattermoe_w1_key} was unexpected but is being created."
                    )

            if len(down_proj_weights) == num_experts:
                LOG.info(f"[{prefix}] Creating {scattermoe_w2_key}")
                # Stack down_proj weights for scattermoe's second ParallelExperts layer
                # Original Qwen3MoeMLP: down_proj.weight (H, I_moe)
                # scattermoe GLUMLP expects: output_experts.weight (E, H, I_moe)
                state_dict[scattermoe_w2_key] = torch.stack(down_proj_weights, dim=0)
                if scattermoe_w2_key in missing_keys:
                    missing_keys.remove(scattermoe_w2_key)
                if scattermoe_w2_key in unexpected_keys:
                    LOG.warning(
                        f"Key {scattermoe_w2_key} was unexpected but is being created."
                    )

            # Clean up
            for key in keys_to_remove_from_state_dict:
                del state_dict[key]
                if key in unexpected_keys:
                    unexpected_keys.remove(key)

            LOG.info(f"[{prefix}] Final missing_keys: {missing_keys}")
            LOG.info(f"[{prefix}] Final unexpected_keys: {unexpected_keys}")
            LOG.info(f"[{prefix}] Error messages: {error_msgs}")

        @staticmethod
        def _state_hook(
            module: nn.Module,
            state_dict: dict,
            prefix: str,
            local_metadata: dict,
        ) -> None:
            if not isinstance(module, ScattermoeGLUBlock):
                return

            num_experts = module.config.num_experts

            scattermoe_w1_key = prefix + "moe_mlp.experts.weight"
            scattermoe_w2_key = prefix + "moe_mlp.output_experts.weight"

            keys_to_remove_from_state_dict = []

            if scattermoe_w1_key in state_dict:
                w1_stacked = state_dict[scattermoe_w1_key]  # Shape: (E, 2*I_moe, H)
                if len(w1_stacked.shape) == 1 and w1_stacked.shape[0] == 0:
                    # hack for empty state dict
                    w1_stacked = torch.zeros(
                        num_experts,
                        0,
                        0,
                        device=w1_stacked.device,
                        dtype=w1_stacked.dtype,
                    )
                keys_to_remove_from_state_dict.append(scattermoe_w1_key)

                # Unstack and split
                for i in range(num_experts):
                    w1_expert_i = w1_stacked[i]  # Shape: (2*I_moe, H)
                    w_up_i, w_gate_i = torch.chunk(
                        w1_expert_i, 2, dim=0
                    )  # Each (I_moe, H)

                    state_dict[prefix + f"experts.{i}.gate_proj.weight"] = w_gate_i
                    state_dict[prefix + f"experts.{i}.up_proj.weight"] = w_up_i

            if scattermoe_w2_key in state_dict:
                w2_stacked = state_dict[scattermoe_w2_key]  # Shape: (E, H, I_moe)
                if len(w2_stacked.shape) == 1 and w2_stacked.shape[0] == 0:
                    # as above, hack for empty state dict
                    w2_stacked = torch.zeros(
                        num_experts,
                        0,
                        0,
                        device=w2_stacked.device,
                        dtype=w1_stacked.dtype,
                    )
                keys_to_remove_from_state_dict.append(scattermoe_w2_key)

                # Unstack
                for i in range(num_experts):
                    w_down_i = w2_stacked[i]  # Shape: (H, I_moe)
                    state_dict[prefix + f"experts.{i}.down_proj.weight"] = w_down_i

            for key in keys_to_remove_from_state_dict:
                del state_dict[key]

    def _decoder_layer_init(
        self: Qwen3MoeDecoderLayer, config: Qwen3MoeConfig, layer_idx: int
    ):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = ScattermoeGLUBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    Qwen3MoeDecoderLayer.__init__ = _decoder_layer_init
    LOG.info("Patched Qwen3MoeDecoderLayer.__init__ to use ScattermoeGLUBlock")
