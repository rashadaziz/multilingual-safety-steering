from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import PreTrainedTokenizerBase
from .hookable import HookableModel
from hooks.caching import build_caching_hook, build_intervention_caching_hook

class InspectableModel(HookableModel):
    _activation_cache: Dict[int, torch.Tensor] = {}
    _intervention_cache: Dict[int, torch.Tensor] = {}

    @property
    def activation_cache(self):
        return torch.stack([acts for _, acts in self._activation_cache.items()])
    
    @property
    def intervention_cache(self):
        return torch.stack([acts for _, acts in self._intervention_cache.items()])

    def cache_activations(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ):  
        self._activation_cache.clear()

        handles = []
        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(build_caching_hook(self._activation_cache, i)))
        
        try:
            self.forward_without_interventions(messages, prompts=prompts, tokenizer=tokenizer, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        return self.activation_cache

    def cache_interventions(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        independent: bool = False,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ):
        self._intervention_cache.clear()

        # temporarily remove original hooks
        for _, handle in self._hooks.items():
            handle.remove()

        # rewrap them
        handles = []
        for i, original_hook in self._hook_wrappers.items():
            handles.append(self.layers[i].register_forward_hook(build_intervention_caching_hook(original_hook, self._intervention_cache, i, independent)))
        
        try:
            self.forward_with_interventions(messages, prompts=prompts, tokenizer=tokenizer, **kwargs)
        finally:
            # remove the re-wrapped versions
            for handle in handles:
                handle.remove()

            # restore the original ones
            for i, original_hook in self._hook_wrappers.items():
                self._hooks[i] = self.layers[i].register_forward_hook(original_hook)
        
        return self.intervention_cache
    
    def forward_without_interventions(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ):
        with self.hooks_disabled():
            self._forward(messages, prompts=prompts, tokenizer=tokenizer, **kwargs)

    def forward_with_interventions(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ):
        self._forward(messages, prompts=prompts, tokenizer=tokenizer, **kwargs)

    def _forward(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ):
        payload: Dict[str, Any] = {}

        active_tokenizer = tokenizer or self.tokenizer
        if prompts is not None and messages is not None:
            raise ValueError("Provide either prompts or messages, not both.")

        if messages is not None:
            if active_tokenizer is None:
                raise ValueError("Tokenizer required to format chat messages.")
            if not messages:
                raise ValueError("messages must contain at least one conversation.")
            first_item = messages[0]
            if isinstance(first_item, Mapping):
                conversations = [messages] 
            else:
                conversations = list(messages)
            prompts_list = [
                active_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]
        elif prompts is not None:
            prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
        else:
            prompts_list = []

        if prompts_list:
            if active_tokenizer is None:
                raise ValueError("Tokenizer required to process prompts.")

            encoded = active_tokenizer(
                prompts_list,
                return_tensors="pt",
                padding=True
            )
            payload.update({k: v.to(self.device) for k, v in encoded.items()})

        if not payload:
            raise ValueError("No inputs were provided to forward.")

        self.model.eval()
        with torch.inference_mode():
            self.model(**payload, use_cache=False, **kwargs)