"""Shared residual-weighting modules for stage-2 training scripts."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class PromptConditionedMLP(nn.Module):
    """A small dense MLP with two hidden layers for prompt conditioning."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc3(hidden_states)
        return hidden_states


class ResidualScalingModelBase(nn.Module):
    """Shared wrapper for frozen Qwen with layer-wise residual scaling hooks."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_layers = self.base_model.config.num_hidden_layers
        self.current_layer_scales: torch.Tensor | None = None
        self._hook_handles = []
        self._freeze_base_model()
        self._register_layer_hooks()

    def _freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def _register_layer_hooks(self) -> None:
        for layer_index, decoder_layer in enumerate(self.base_model.model.layers):
            handle = decoder_layer.register_forward_hook(
                self._make_layer_hook(layer_index),
            )
            self._hook_handles.append(handle)

    def _make_layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            if self.current_layer_scales is None:
                return output

            residual_input = inputs[0]
            layer_scales = self.current_layer_scales
            if layer_scales.dim() == 1:
                scale = layer_scales[layer_index].view(1, 1, 1)
            else:
                scale = layer_scales[:, layer_index].view(layer_scales.shape[0], 1, 1)

            scale = scale.to(device=output.device, dtype=output.dtype)
            delta = output - residual_input
            return residual_input + scale * delta

        return hook

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        full_state = self.state_dict()
        return {
            name: tensor.detach().cpu()
            for name, tensor in full_state.items()
            if name in trainable_names
        }

    def _forward_with_scales(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer_scales: torch.Tensor,
    ):
        self.current_layer_scales = layer_scales
        try:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        finally:
            self.current_layer_scales = None


class StaticResidualModel(ResidualScalingModelBase):
    """Frozen base model with one learned residual scale per decoder layer."""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__(base_model=base_model)
        self.raw_layer_scales = nn.Parameter(torch.zeros(self.num_layers))

    def get_layer_scales(self) -> torch.Tensor:
        return 1.0 + torch.tanh(self.raw_layer_scales)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: dict,
    ):
        return self._forward_with_scales(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_scales=self.get_layer_scales(),
        )


class PromptConditionedResidualModel(ResidualScalingModelBase):
    """Frozen base model with prompt-conditioned residual scales."""

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        mlp_hidden_size: int,
        rms_norm_eps: float | None = None,
    ) -> None:
        super().__init__(base_model=base_model)
        del rms_norm_eps
        self.prompt_hidden_cache: dict[str, torch.Tensor] = {}
        self.conditioner = PromptConditionedMLP(
            input_size=self.base_model.config.hidden_size,
            hidden_size=mlp_hidden_size,
            output_size=self.num_layers,
        )

    def get_prompt_hidden(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        if example_id is None:
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    use_cache=False,
                )
            return prompt_outputs.last_hidden_state[:, -1, :]

        cached_hidden_states: list[torch.Tensor | None] = [None] * len(example_id)
        missing_indices: list[int] = []
        for index, current_example_id in enumerate(example_id):
            cached_hidden = self.prompt_hidden_cache.get(current_example_id)
            if cached_hidden is None:
                missing_indices.append(index)
            else:
                cached_hidden_states[index] = cached_hidden

        if missing_indices:
            missing_index_tensor = torch.tensor(
                missing_indices,
                device=prompt_input_ids.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                prompt_outputs = self.base_model.model(
                    input_ids=prompt_input_ids.index_select(0, missing_index_tensor),
                    attention_mask=(
                        None
                        if prompt_attention_mask is None
                        else prompt_attention_mask.index_select(0, missing_index_tensor)
                    ),
                    use_cache=False,
                )
            missing_hidden_states = prompt_outputs.last_hidden_state[:, -1, :].detach().cpu()
            for local_index, batch_index in enumerate(missing_indices):
                current_example_id = example_id[batch_index]
                cached_hidden = missing_hidden_states[local_index].clone()
                self.prompt_hidden_cache[current_example_id] = cached_hidden
                cached_hidden_states[batch_index] = cached_hidden

        stacked_hidden_states = torch.stack(cached_hidden_states, dim=0)
        return stacked_hidden_states.to(
            device=prompt_input_ids.device,
            dtype=self.conditioner.fc1.weight.dtype,
        )

    def get_layer_scales(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
    ) -> torch.Tensor:
        prompt_hidden = self.get_prompt_hidden(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        raw_scales = self.conditioner(prompt_hidden)
        return 1.0 + 0.1 * raw_scales

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        example_id: list[str] | None = None,
        **_: dict,
    ):
        layer_scales = self.get_layer_scales(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            example_id=example_id,
        )
        return self._forward_with_scales(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            layer_scales=layer_scales,
        )
