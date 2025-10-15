import torch as t

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