import torch as t
from tqdm.auto import tqdm
from core.inspectable import InspectableModel
from data import load_flores_plus, FLORES_LANGUAGE_CODES

def compute_language_means(model: InspectableModel, batch_size: int = 32):
    if model.tokenizer is None:
        raise ValueError("Model tokenizer must be provided to compute language means.")

    prompts = {
        lang: load_flores_plus(lang) for lang in FLORES_LANGUAGE_CODES
    }

    language_means: list[t.Tensor] = []
    languages = list(prompts.keys())

    for lang in tqdm(languages, desc="Languages"):
        dataset = prompts[lang]

        sum_hidden: t.Tensor | None = None
        running_count = 0

        batch_iterator = range(0, len(dataset), batch_size)
        for i in tqdm(
            batch_iterator,
            desc=f"{lang.value} batches",
            leave=False,
        ):
            batch = dataset[i : i + batch_size]
            conversations = [
                [{"role": "user", "content": example.prompt}] for example in batch
            ]

            activations = model.cache_activations(conversations)  # (num_layers, batch, seq_len, hidden_size)

            if sum_hidden is None:
                num_layers, _, _, hidden_size = activations.shape
                sum_hidden = t.zeros(
                    (num_layers, hidden_size),
                    dtype=t.float64,
                )

            for layer_idx in range(activations.shape[0]):
                layer_hidden = activations[layer_idx]  # (batch, seq_len, hidden_size)
                final_tokens = layer_hidden[:, -1, :]  # left padding keeps final token at -1
                sum_hidden[layer_idx] += final_tokens.to(t.float64).sum(dim=0)

            running_count += len(batch)

        if sum_hidden is None or running_count == 0:
            raise ValueError(f"Unable to compute language means for language '{lang.value}'.")

        mean_hidden = (sum_hidden / running_count).to(dtype=t.float32).contiguous()
        language_means.append(mean_hidden)

    # (num_langs, num_layers, hidden_size) => (num_layers, hidden_size, num_langs)
    return t.stack(language_means, dim=0).permute(1, 2, 0).contiguous()

def compute_language_subspace(language_means: t.Tensor, rank: int) -> tuple[t.Tensor, t.Tensor]:
    """
    This function computes the language subspace based on the work of Xie et al. (https://arxiv.org/pdf/2401.05792)

    Args:
        language_means: A `torch.Tensor` of language embedding means with shape `(d, L)` where `d` is the embedding dimension, and `L` is the number of languages
        rank: Corresponds to what rank the final `M_specific` should be i.e how many directions/axes are kept in the final susbpace

    Returns:
        (M_specific, Gamma):
            - *M_specific*: `torch.Tensor` of shape `(d, r)` basis for language-specific subspace.
            - *Gamma*: `torch.Tensor` of shape `(L, r)` coordinates of each language mean in that subspace.
    """
    M = language_means
    if M.ndim != 2:
        raise ValueError("language_means must be 2D (d, L).")
    d, L = M.shape
    if not (1 <= rank <= L):
        raise ValueError(f"rank must be in [1, {L}], got {rank}.")

    # mean direction
    mean = M.mean(dim=1, keepdim=True) # (d, 1)
    centered = M - mean # (d, L)

    # Approximate M in low rank
    U, S, Vh = t.linalg.svd(centered, full_matrices=False)
    M_specific: t.Tensor = U[:, :rank] # (d, r)  (candidate M_specific)
    Gamma: t.Tensor = Vh.T[:rank, :] @ t.diag(S[:rank]) # (L, r)  (candidate Gamma)

    ones = t.ones(L, 1, dtype=M.dtype, device=M.device)
    M_reconstructed = mean + M_specific @ Gamma.T # (d, L)
    M_inv = t.linalg.pinv(M_reconstructed.T)
    mu = M_inv @ ones
    mu = mu / (mu.norm() + 1e-12)

    # Force orthogonality
    centered_new = M_reconstructed - mu @ ones.T
    U2, S2, Vh2 = t.linalg.svd(centered_new, full_matrices=False)

    return U2[:, :rank], Vh2[:rank, :].T @ t.diag(S2[:rank])
