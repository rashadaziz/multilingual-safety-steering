from contextlib import contextmanager
from typing import Any, Dict

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
        self.layers: list[torch.nn.Module] = tuple(self.model.model.layers)
        self._hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        self._hook_wrappers: Dict[int, Any] = {}
        self._hooks_enabled = True

    def register_hook(self, layer_index: int, hook_fn) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer_index {layer_index} is out of bounds for {len(self.layers)} layers.")
        if layer_index in self._hooks:
            raise ValueError(f"A hook is already registered for layer {layer_index}.")

        def wrapped(module, inputs, output):
            if self._hooks_enabled:
                return hook_fn(module, inputs, output)

        handle = self.layers[layer_index].register_forward_hook(wrapped)
        self._hooks[layer_index] = handle
        self._hook_wrappers[layer_index] = wrapped

    def remove_hook(self, layer_index: int) -> None:
        handle = self._hooks.pop(layer_index, None)
        self._hook_wrappers.pop(layer_index, None)
        if handle is not None:
            handle.remove()

    def clear_hooks(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
        self._hook_wrappers.clear()

    @contextmanager
    def hooks_disabled(self):
        prev = self._hooks_enabled
        self._hooks_enabled = False
        try:
            yield
        finally:
            self._hooks_enabled = prev