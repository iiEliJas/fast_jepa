import torch


def embedding_diagnostics(z):
    # z: (N, K)
    std_per_dim = z.std(dim=0)
    # effective rank via SVD of centered
    zc = z - z.mean(dim=0)
    s = torch.linalg.svdvals(zc)
    # effective rank (Roy & Vetterli)
    p = s / s.sum()
    erank = torch.exp(-(p * p.log()).sum())
    return {"std_mean": std_per_dim.mean().item(), "erank": erank.item()}

