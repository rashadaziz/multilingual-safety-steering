from typing import Optional
import torch

def build_refusal_hook(refusal_vector: torch.Tensor, vector_strength: float = 0, ablation_projection: Optional[torch.Tensor] = None, ablation_strength: float = 0) -> callable:
    """
    Builds a hook that adds a refusal vector to the activations.

    Args:
        refusal_vector (torch.Tensor): The refusal direction vector.
        vector_strength (float): The strength with which to add the refusal vector.
        ablation_projection (Optional[torch.Tensor]): The ablation projection vector.
        ablation_strength (float): The strength of the ablation.

    Returns:
        function: A hook function to be registered.
    """
    def _hook(_module, _inputs, outputs):
        if vector_strength == 0:
            return outputs
        
        # Ensure the refusal vector is on the same device and dtype as outputs
        rv = refusal_vector.to(device=outputs.device, dtype=outputs.dtype)

        if ablation_projection is not None and ablation_strength > 0:
            ap = ablation_projection.to(device=outputs.device, dtype=outputs.dtype)
            # Project refusal vector to be orthogonal to ablation projection
            coords = rv @ ap
            projection = coords @ ap.T
            rv = rv - ablation_strength * projection
            # Normalize the adjusted refusal vector
            rv = rv / rv.norm()

        # rv should have shape (hidden_size,) which will broadcast correctly
        return outputs + vector_strength * rv

    return _hook