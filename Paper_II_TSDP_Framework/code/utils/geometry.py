import torch
import numpy as np

def compute_cosine_similarity_matrix(v_matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine similarity (c) for a given V-space matrix.
    Args:
        v_matrix: (seq_len, head_dim)
    Returns:
        c_matrix: (seq_len, seq_len)
    """
    dot_v = torch.matmul(v_matrix, v_matrix.T)
    norm_v = torch.norm(v_matrix, dim=-1, keepdim=True)
    c_matrix = torch.clamp(dot_v / (norm_v * norm_v.T + 1e-8), -1.0, 1.0)
    return c_matrix

def compute_s_variance(c_matrix: torch.Tensor) -> float:
    """
    Computes the orthogonal redundancy variance (Var(s)).
    s = sqrt(1 - c^2)
    """
    s_matrix = torch.sqrt(1 - c_matrix**2 + 1e-8)
    return s_matrix.var().item()

def compute_pc1_ratio(matrix: torch.Tensor) -> float:
    """
    Computes the first principal component (PC1) ratio.
    Args:
        matrix: (N, dim) V-space representations.
    """
    if matrix.shape[0] < 2:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    # SVD decomposition
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    explained_variance = S ** 2
    return (explained_variance[0] / explained_variance.sum()).item()

def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """
    Computes KL divergence between two logit distributions.
    """
    p = torch.log_softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    return torch.nn.functional.kl_div(p, q, reduction='sum').item()