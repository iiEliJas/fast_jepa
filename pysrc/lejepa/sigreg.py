import torch
from torch import nn


def make_sigreg(in_dim, num_slices=256, t_max=3.0, n_knots=17):
    return SlicedEppsPulley(in_dim, num_slices, t_max, n_knots)


class SlicedEppsPulley(nn.Module):
    def __init__(self, in_dim, num_slices=256, t_max=3.0, n_knots=17):
        super().__init__()
        assert n_knots % 2 == 1
        self.in_dim = in_dim
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t * t)
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("phi", phi)

    def forward(self, x, A=None):
        # x: (N, K) or (V, N, K)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False
        V, N, K = x.shape
        # shared directions across views
        if A is None:
            A = torch.randn(K, self.num_slices, device=x.device, dtype=x.dtype)
        A = A / A.norm(dim=0, keepdim=True)
        # project: (V, N, S)
        proj = torch.einsum("vnk,ks->vns", x, A)
        x_t = proj.unsqueeze(-1) * self.t  # (V, N, S, T)
        cos_mean = torch.cos(x_t).mean(dim=1)  # (V, S, T)
        sin_mean = torch.sin(x_t).mean(dim=1)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        per_view = (err @ self.weights) * N  # (V, S)
        per_view = per_view.mean(dim=1)  # (V,)
        out = per_view.mean()  # scalar
        if single:
            pass
        return out
