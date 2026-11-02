try:
    import lejepa  # official package if installed

    HAS_OFFICIAL = True
except Exception:  # pragma: no cover
    lejepa = None
    HAS_OFFICIAL = False


def official_sigreg(*args, **kwargs):
    if not HAS_OFFICIAL:
        raise RuntimeError("official lejepa package not installed")
    return lejepa.multivariate.SlicingUnivariateTest(*args, **kwargs)
