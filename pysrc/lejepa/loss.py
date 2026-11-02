def lejepa_loss(z, sigreg, lamb=0.02, pred_reduction="mean_over_k"):
    # z: (V, N, K) projector outputs; first V_g global views
    V, N, K = z.shape
    # mu over global views (default all views for parity recipe)
    mu = z.mean(dim=0)  # (N, K) if all views; paper uses global only, config controls
    # L_pred mean over all elements
    pred = (mu.unsqueeze(0) - z) ** 2
    if pred_reduction == "mean_over_k":
        l_pred = pred.mean()
    else:
        l_pred = pred.sum(dim=-1).mean()
    # SIGReg per view
    l_sig = 0.0
    for v in range(V):
        l_sig = l_sig + sigreg(z[v])
    l_sig = l_sig / V
    return (1 - lamb) * l_pred + lamb * l_sig, l_pred.detach(), l_sig.detach()
