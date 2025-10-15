from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import PreTrainedTokenizerBase
from .hookable import HookableModel

class SteerableModel(HookableModel):
    def generate(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        prompts: str | Sequence[str] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **generate_kwargs: Any,
    ) -> list[str]:
        active_tokenizer = tokenizer or self.tokenizer
        if active_tokenizer is None:
            raise ValueError("A tokenizer must be provided to generate text.")
        if prompts is not None and messages is not None:
            raise ValueError("Provide either prompts or messages, not both.")

        prompt_texts: list[str]
        if messages is not None:
            if not messages:
                raise ValueError("messages must contain at least one conversation.")
            first_item = messages[0]
            if isinstance(first_item, Mapping):
                conversations = [messages]
            else:
                conversations = list(messages)
            prompt_texts = [
                active_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]
        elif prompts is not None:
            prompt_texts = [prompts] if isinstance(prompts, str) else list(prompts)
        else:
            raise ValueError("Either prompts or messages must be provided.")

        encoded = active_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        input_ids = encoded["input_ids"]
        if "attention_mask" in encoded:
            prompt_lengths = encoded["attention_mask"].sum(dim=1).to(torch.long)
        else:
            prompt_lengths = torch.tensor(
                [input_ids.shape[1]] * input_ids.shape[0],
                device=input_ids.device,
                dtype=torch.long,
            )

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(**encoded, **generate_kwargs)

        continuations: list[str] = []
        for i in range(outputs.shape[0]):
            start = int(prompt_lengths[i].item())
            continuation_tokens = outputs[i, start:]
            if continuation_tokens.numel() == 0:
                continuations.append("")
                continue
            token_ids = continuation_tokens.detach().cpu().tolist()
            continuations.append(active_tokenizer.decode(token_ids, skip_special_tokens=True))
        return continuations