def lejepa_loss(z, sigreg, lamb=0.02, pred_reduction="mean_over_k"):
    # z: (V, N, K) projector outputs; all views are used for mu in the parity
    # recipe (V_g = V). See protocol section 5.1.
    if z.dim() != 3:
        raise ValueError(f"expected z shape (V, N, K), got {tuple(z.shape)}")
    V, N, K = z.shape
    if V == 0 or N == 0:
        raise ValueError("lejepa_loss needs at least one view and one sample")
    if not 0.0 <= lamb <= 1.0:
        raise ValueError(f"lamb must be in [0, 1], got {lamb}")

    mu = z.mean(dim=0)  # (N, K)
    pred = (mu.unsqueeze(0) - z) ** 2
    if pred_reduction == "mean_over_k":
        l_pred = pred.mean()
    elif pred_reduction == "sum_over_k":
        l_pred = pred.sum(dim=-1).mean()
    else:
        raise ValueError(f"unknown pred_reduction={pred_reduction!r}")

    # One call on (V, N, K): all views share the same random directions
    # (protocol 5.1/5.2). The module already averages over views.
    l_sig = sigreg(z)
    return (1 - lamb) * l_pred + lamb * l_sig, l_pred.detach(), l_sig.detach()
