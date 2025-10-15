from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class HookableModel:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.layers = tuple(self.model.modules())
        self._hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}

    def register_hook(self, layer_index: int, hook_fn) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer_index {layer_index} is out of bounds for {len(self.layers)} layers.")
        if layer_index in self._hooks:
            raise ValueError(f"A hook is already registered for layer {layer_index}.")
        handle = self.layers[layer_index].register_forward_hook(hook_fn)
        self._hooks[layer_index] = handle

    def remove_hook(self, layer_index: int) -> None:
        handle = self._hooks.pop(layer_index, None)
        if handle is not None:
            handle.remove()

    def clear_hooks(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()