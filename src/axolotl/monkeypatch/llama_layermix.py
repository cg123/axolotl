# pylint: skip-file
import copy
from typing import List

import torch
import torch.nn.utils.parametrize as parameterize
import transformers
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True


class LinearMixtureParameterization(nn.Module):
    def __init__(self, in_features: int, out_features: int, donors: List[nn.Linear]):
        super(LinearMixtureParameterization, self).__init__()

        self.donors = donors
        self.scales = nn.Parameter(torch.ones(len(donors)))

    def forward(self, *args):
        weights = torch.zeros_like(self.donors[0].weight)
        for idx in range(len(self.donors)):
            weights += self.donors[idx].weight * self.scales[idx]
        return weights / len(self.donors)


def make_mixture_linear(target: nn.Linear, donors: List[nn.Parameter]):
    assert all(
        donor.weight.shape == target.weight.shape for donor in donors
    ), "Donor sizes must match target"

    (out_features, in_features) = target.weight.shape
    target.weight.requires_grad_(False)
    parameterize.register_parametrization(
        target,
        "weight",
        LinearMixtureParameterization(in_features, out_features, donors),
    )


class LinearLayerMixtureLlama(LlamaForCausalLM):
    def __init__(self, donor: LlamaForCausalLM, new_layer_count: int):
        new_config: LlamaConfig = copy.deepcopy(donor.config)
        new_config.num_hidden_layers = new_layer_count

        with NoInit():
            super(LinearLayerMixtureLlama, self).__init__(new_config)

        self.donor = donor.requires_grad_(False)
        with torch.no_grad():
            self.model.embed_tokens.weight[:, :] = donor.model.embed_tokens.weight.data
            self.model.norm.weight[:] = donor.model.norm.weight

        for idx in range(len(self.model.layers)):
            layer: LlamaDecoderLayer = self.model.layers[idx]

            for mlp_attr in ["up_proj", "gate_proj", "down_proj"]:
                make_mixture_linear(
                    getattr(layer.mlp, mlp_attr),
                    [getattr(dl.mlp, mlp_attr) for dl in donor.model.layers],
                )
            for attn_attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                make_mixture_linear(
                    getattr(layer.self_attn, attn_attr),
                    [getattr(dl.self_attn, attn_attr) for dl in donor.model.layers],
                )

            with torch.no_grad():
                layer.input_layernorm.weight[:] = donor.model.layers[
                    idx
                ].input_layernorm.weight.data
                layer.post_attention_layernorm.weight[:] = donor.model.layers[
                    idx
                ].post_attention_layernorm.weight.data

        with torch.no_grad():
            self.lm_head.weight[:, :] = donor.lm_head.weight.data
