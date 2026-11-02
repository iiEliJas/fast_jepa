import torch


def embedding_diagnostics(z):
    # z: (N, K)
    if z.dim() != 2:
        raise ValueError(f"expected z shape (N, K), got {tuple(z.shape)}")
    if z.numel() == 0:
        return {
            "std_mean": 0.0,
            "std_min": 0.0,
            "std_max": 0.0,
            "erank": 0.0,
            "collapsed": True,
        }

    # Population std avoids NaN for N == 1.
    std_per_dim = z.std(dim=0, unbiased=False)
    zc = z - z.mean(dim=0)
    s = torch.linalg.svdvals(zc)
    total = s.sum()
    if not torch.isfinite(total) or total <= 0:
        erank_val = 0.0
    else:
        p = s / total
        # entr(0) == 0, so zero singular values do not produce NaN.
        erank_val = torch.exp(torch.special.entr(p).sum()).item()

    std_mean = std_per_dim.mean().item()
    collapsed = bool(std_mean < 1e-6 or erank_val < 1.0 + 1e-6)
    return {
        "std_mean": std_mean,
        "std_min": std_per_dim.min().item(),
        "std_max": std_per_dim.max().item(),
        "erank": erank_val,
        "collapsed": collapsed,
    }
