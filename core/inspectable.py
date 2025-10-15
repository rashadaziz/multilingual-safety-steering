from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import PreTrainedTokenizerBase
from .hookable import HookableModel

class InspectableModel(HookableModel):
    def forward(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        use_cache: bool = False,
        inputs: Mapping[str, Any] | None = None,
        **model_inputs: Any,
    ):
        payload: Dict[str, Any] = {}
        if inputs is not None:
            payload.update(dict(inputs))

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

        payload.update(model_inputs)
        if not payload:
            raise ValueError("No inputs were provided to forward.")

        self.model.eval()
        with torch.no_grad():
            self.model(**payload, use_cache=use_cache)