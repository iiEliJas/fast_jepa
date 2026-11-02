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

    def forward(self, x, A=None, return_per_slice=False):
        # x: (N, K) or (V, N, K). Returns a scalar averaged over views and slices.
        # return_per_slice=True returns the (S,) statistic averaged over views,
        # which the parity tests compare against the official EppsPulley.
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"expected 2D or 3D input, got shape {tuple(x.shape)}")
        V, N, K = x.shape
        if K != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got K={K}")
        if N == 0:
            raise ValueError("SIGReg needs at least one sample per view")

        # FP64 is only needed for gradcheck; FP16/BF16 statistics stay FP32
        # The projection may still run in x.dtype.
        stat_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        t = self.t.to(stat_dtype)
        weights = self.weights.to(stat_dtype)
        phi = self.phi.to(stat_dtype)

        # Sample and normalize directions in FP32 (FP64 for gradcheck), then
        # cast to the projection dtype. One A is shared by all views.
        dir_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if A is None:
            A = torch.randn(K, self.num_slices, device=x.device, dtype=dir_dtype)
        else:
            if A.shape != (K, self.num_slices):
                raise ValueError(f"expected A shape {(K, self.num_slices)}, got {tuple(A.shape)}")
            A = A.to(device=x.device, dtype=dir_dtype)
        A = (A / A.norm(dim=0, keepdim=True)).detach().to(x.dtype)

        proj = torch.einsum("vnk,ks->vns", x, A)  # (V, N, S)
        x_t = proj.unsqueeze(-1).to(stat_dtype) * t  # (V, N, S, T)
        cos_mean = torch.cos(x_t).mean(dim=1)  # (V, S, T)
        sin_mean = torch.sin(x_t).mean(dim=1)
        err = (cos_mean - phi).square() + sin_mean.square()
        per_view = (err @ weights) * N  # (V, S)
        per_slice = per_view.mean(dim=0)  # (S,), mean over views
        if return_per_slice:
            return per_slice
        return per_slice.mean()  # mean over views and slices
