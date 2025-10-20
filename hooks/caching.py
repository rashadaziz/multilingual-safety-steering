import torch

from typing import Dict, Callable

def build_caching_hook(cache: Dict[int, torch.Tensor], layer: int):
    def _hook(_module, _inputs, outputs: torch.Tensor):
        cache[layer] = outputs.detach().cpu()
    
    return _hook

def build_intervention_caching_hook(intervention_fn: Callable[..., torch.Tensor], cache: Dict[int, torch.Tensor], layer: int, independent: bool):
    def _hook(_module, _inputs, outputs: torch.Tensor):
        intervened_outputs = intervention_fn(_module, _inputs, outputs)
        cache[layer] = intervened_outputs.detach().cpu()

        if not independent:
            return intervened_outputs
    
    return _hook