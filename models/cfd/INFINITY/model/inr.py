import torch
import torch.nn as nn

from .fourier_features import FourierFeatures


class ModulatedINR(nn.Module):
    """Modulated Implicit Neural Representation (Fourier Features + FiLM shift).

    f_{theta,phi}(x) = W_L( chi_{L-1} o ... o chi_0(x) ) + b_L
    chi_j(eta_j) = sigma( W_j eta_j + b_j + phi_j )
    eta_0 = gamma(x) (Fourier features). phi = (phi_0, ..., phi_{L-1}) shift terms.
    theta = shared weights (W_j, b_j); phi is per-example modulation.
    """

    def __init__(self, coord_dim=2, fourier_dim=32, hidden_dim=256, n_layers=4,
                 latent_dim=32, activation="relu", out_dim=1):
        super().__init__()
        self.coord_dim = coord_dim
        self.fourier_dim = fourier_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.fourier = FourierFeatures(coord_dim, fourier_dim)

        # number of hidden layers between Fourier input and output
        n_hidden = n_layers - 1  # includes input->hidden and hidden->...->output
        # layers: input(fourier) -> hidden (n_hidden-1 times) -> output(out_dim)
        self.linears = nn.ModuleList()
        self.n_params = []
        prev = 2 * fourier_dim
        for _ in range(n_hidden - 1):
            lin = nn.Linear(prev, hidden_dim)
            self.linears.append(lin)
            self.n_params.append(hidden_dim)
            prev = hidden_dim
        # final output layer -> out_dim
        out_lin = nn.Linear(prev, out_dim)
        self.linears.append(out_lin)
        self.n_params.append(out_dim)

        # total FiLM shift parameters = sum of bias sizes per hidden layer
        self.phi_dim = sum(self.n_params)
        # hypernetwork maps latent code -> phi
        self.hyper = nn.Linear(latent_dim, self.phi_dim)

        act = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[activation]
        self.activation = act()

    def hypernet(self, z):
        # z: (B, latent_dim) -> phi: (B, phi_dim)
        return self.hyper(z)

    def forward(self, x, phi):
        """x: (N, coord_dim); phi: (phi_dim,) or (B, phi_dim)."""
        feats = self.fourier(x)  # (N, 2*fourier_dim)
        h = feats
        phi_chunks = torch.split(phi, self.n_params, dim=-1)
        for i, lin in enumerate(self.linears):
            shift = phi_chunks[i]
            # broadcast shift over nodes: shift (out_dim,) or (B, out_dim)
            if shift.dim() == 1:
                shift = shift.unsqueeze(0)  # (1, out_dim)
            # now shift (1, out_dim) broadcasts with h (N, out_dim) -> (N, out_dim)
            if i < len(self.linears) - 1:
                h = self.activation(lin(h) + shift)
            else:
                h = lin(h) + shift
        return h  # (N, out_dim)
