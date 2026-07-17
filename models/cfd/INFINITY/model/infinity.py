import torch
import torch.nn as nn

from .inr import ModulatedINR

FUNCTIONS = ["d", "n", "vx", "vy", "p", "nut"]
INPUT_FUNCTIONS = ["d", "n"]
OUTPUT_FUNCTIONS = ["vx", "vy", "p", "nut"]


class INFINITY(nn.Module):
    """INFINITY model: 6 modulated INRs + mapping network g_psi.

    - 2 input INRs (d volume, n surface) encode geometry to latent codes.
    - 4 output INRs decode latent codes to physical fields.
    - g_psi maps (z_d, z_n, Vx, Vy) -> (z_vx, z_vy, z_p, z_nut).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg["inr"]["latent_dim"]
        self.K = cfg["inr_train"]["K_inner"]

        inr_kwargs = dict(
            coord_dim=cfg["inr"]["coord_dim"],
            fourier_dim=cfg["inr"]["fourier_dim"],
            hidden_dim=cfg["inr"]["hidden_dim"],
            n_layers=cfg["inr"]["n_layers"],
            latent_dim=self.latent_dim,
            activation=cfg["inr"]["activation"],
        )
        self.inrs = nn.ModuleDict({u: ModulatedINR(**inr_kwargs, out_dim=2 if u == "n" else 1)
                                    for u in FUNCTIONS})

        # mapping network g_psi: (2*latent + 2) -> 4*latent
        mh = cfg["mapping"]["hidden_dim"]
        ml = cfg["mapping"]["n_layers"]
        gin = 2 * self.latent_dim + 2
        gout = 4 * self.latent_dim
        layers = [nn.Linear(gin, mh), nn.ReLU()]
        for _ in range(ml - 2):
            layers += [nn.Linear(mh, mh), nn.ReLU()]
        layers += [nn.Linear(mh, gout)]
        self.g_psi = nn.Sequential(*layers)

    # ---- encoding (auto-decoding, Eq 1) ----
    def encode(self, inr, coords, values, inner_lr, K=None):
        """First-order auto-encoding of a field into latent code z."""
        K = K or self.K
        device = next(inr.parameters()).device
        z = torch.zeros(self.latent_dim, device=device, requires_grad=True)
        for _ in range(K):
            phi = inr.hypernet(z.unsqueeze(0))[0]
            pred = inr.forward(coords.to(device), phi)
            l = ((pred - values.to(device)) ** 2).mean()
            g = torch.autograd.grad(l, z, create_graph=False)[0]
            with torch.no_grad():
                z = z - inner_lr * g
            z.requires_grad_(True)
        return z.detach()

    def decode_field(self, inr, coords, z):
        phi = inr.hypernet(z.unsqueeze(0))[0]
        out = inr.forward(coords, phi)  # (N, out_dim)
        return out.squeeze(-1)  # (N,) for out_dim=1

    # ---- process ----
    def process(self, z_d, z_n, Vx, Vy):
        """z_d,z_n: (latent_dim,); Vx,Vy: scalars -> 4 output codes (4*latent,)."""
        device = next(self.g_psi.parameters()).device
        cond = torch.cat([z_d.to(device), z_n.to(device),
                          torch.tensor([Vx], device=device, dtype=torch.float32),
                          torch.tensor([Vy], device=device, dtype=torch.float32)])
        return self.g_psi(cond)  # (4*latent,)

    def predict_fields(self, coords, z_out_codes):
        """coords: (N,2); z_out_codes: (4*latent,) -> (N,4) fields vx,vy,p,nut."""
        codes = torch.chunk(z_out_codes, 4, dim=-1)  # 4 x (latent,)
        out = []
        for i, u in enumerate(OUTPUT_FUNCTIONS):
            out.append(self.decode_field(self.inrs[u], coords, codes[i]))
        return torch.stack(out, dim=-1)  # (N,4) per node
