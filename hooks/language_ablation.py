import torch

def build_language_ablation_hook(language_subspace: torch.Tensor, ablation_strength: float = 1.0):
    def _hook(_module, _inputs, outputs: torch.Tensor):
        # outputs: (batch_size, seq_len, hidden_size)
        if outputs.ndim != 3 or outputs.shape[1] == 0:
            return outputs

        final_token = outputs[:, -1, :]  # final token remains at -1 with left padding
        coords = final_token @ language_subspace  # (batch_size, rank)
        proj = coords @ language_subspace.T  # (batch_size, hidden_size)
        ablated_final = final_token - ablation_strength * proj

        modified = outputs.clone()
        modified[:, -1, :] = ablated_final
        return modified

    return _hook
