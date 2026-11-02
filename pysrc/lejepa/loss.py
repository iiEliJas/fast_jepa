def lejepa_loss(
    z,
    sigreg,
    lamb=0.02,
    pred_reduction="mean_over_k",
    V_g=None,
    sigreg_mode="per_view",
):
    """LeJEPA objective (protocol section 5.1).

    z: (V, N, K) projector outputs, views ordered so that the first V_g are the
    global views. L_pred averages the squared distance to the mean of the global
    views over all (view, image, coordinate) elements. SIGReg is computed once
    per step so every view shares the same random directions.

    ``sigreg_mode``:
      - ``"per_view"``: SIGReg over the N images of each view, averaged over
        views (paper / official minimal example).
      - ``"joint"``: one SIGReg over all V*N embeddings (stable_pretraining).
    """
    if z.dim() != 3:
        raise ValueError(f"expected z shape (V, N, K), got {tuple(z.shape)}")
    V, N, K = z.shape
    if V == 0 or N == 0:
        raise ValueError("lejepa_loss needs at least one view and one sample")
    if not 0.0 <= lamb <= 1.0:
        raise ValueError(f"lamb must be in [0, 1], got {lamb}")
    if V_g is None:
        V_g = V
    if not 1 <= V_g <= V:
        raise ValueError(f"V_g must be in [1, {V}], got {V_g}")

    mu = z[:V_g].mean(dim=0)  # (N, K)
    pred = (mu.unsqueeze(0) - z) ** 2
    if pred_reduction == "mean_over_k":
        l_pred = pred.mean()
    elif pred_reduction == "sum_over_k":
        l_pred = pred.sum(dim=-1).mean()
    else:
        raise ValueError(f"unknown pred_reduction={pred_reduction!r}")

    if sigreg_mode == "per_view":
        # One call on (V, N, K): all views share the same random directions.
        l_sig = sigreg(z)
    elif sigreg_mode == "joint":
        l_sig = sigreg(z.reshape(V * N, K))
    else:
        raise ValueError(f"unknown sigreg_mode={sigreg_mode!r}")
    return (1 - lamb) * l_pred + lamb * l_sig, l_pred.detach(), l_sig.detach()
