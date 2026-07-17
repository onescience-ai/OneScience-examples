import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.config import load_config
from scripts.data.airfrans_dataset import AirfRANSDataPipe, build_normalizer, apply_normalizer
from scripts.data.sampling import sample_volume_points
from model.infinity import INFINITY, OUTPUT_FUNCTIONS, INPUT_FUNCTIONS
from scripts.losses import code_mse_loss


def main(config_path):
    cfg = load_config(config_path)
    np.random.seed(cfg["defaults"]["seed"]); torch.manual_seed(cfg["defaults"]["seed"])

    with open(cfg["defaults"]["manifest"]) as f:
        m = json.load(f)
    pipe = AirfRANSDataPipe(cfg["defaults"]["data_root"], m[cfg["defaults"]["train_split"]])
    raw = [pipe[i] for i in range(len(pipe))]
    norm = build_normalizer(raw)
    cases = [apply_normalizer(c, norm) for c in raw]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = INFINITY(cfg).to(device)
    inr_ckpt = torch.load(os.path.join(cfg["defaults"]["out_dir"], "infinity_inr.pt"),
                          map_location=device, weights_only=False)
    model.inrs.load_state_dict(inr_ckpt["inr_state"])
    model.inrs.requires_grad_(False)

    K = cfg["inr_train"]["K_inner"]; inner_lr = cfg["inr_train"]["inner_lr"]
    n_pts = cfg["sampling"]["n_pts"]; ld = cfg["inr"]["latent_dim"]
    surface_bias = cfg["sampling"].get("surface_bias", 0.0)
    rng = np.random.RandomState(cfg["defaults"]["seed"])

    # Encode all cases to z-codes (first-order)
    def _enc(inr, coords, vals):
        z = torch.zeros(ld, device=device, requires_grad=True)
        for _ in range(K):
            phi = inr.hypernet(z.unsqueeze(0))[0]; pred = inr.forward(coords, phi)
            g = torch.autograd.grad(((pred - vals)**2).mean(), z, create_graph=False)[0]
            with torch.no_grad():
                z = z - inner_lr * g
            z.requires_grad_(True)
        return z.detach()

    encodings = []
    for case in cases:
        vidx = sample_volume_points(case["x"], case["d"], n_pts, surface_bias, rng)
        cx = torch.tensor(case["x"][vidx], dtype=torch.float32, device=device)
        csx = torch.tensor(case["surf_x"][:n_pts], dtype=torch.float32, device=device)
        cn = torch.tensor(np.concatenate([case["nx"], case["ny"]], 1)[:n_pts], dtype=torch.float32, device=device)
        zd = _enc(model.inrs["d"], cx, torch.tensor(case["d"][vidx, 0], dtype=torch.float32, device=device))
        zn = _enc(model.inrs["n"], csx, cn)

        zouts = []
        for u in OUTPUT_FUNCTIONS:
            col = {"vx": 0, "vy": 1, "p": 2, "nut": 3}[u]
            zc = _enc(model.inrs[u], cx, torch.tensor(case["y"][:n_pts, col], dtype=torch.float32, device=device))
            zouts.append(zc)

        zouts = torch.stack(zouts).reshape(-1)
        cond = torch.cat([zd, zn,
                          torch.tensor([case["Vx"]], device=device),
                          torch.tensor([case["Vy"]], device=device)])
        encodings.append((cond, zouts))

    opt = torch.optim.Adam(model.g_psi.parameters(), lr=cfg["mapping"]["lr"])
    bs = cfg["mapping"]["batch_size"]
    epochs = cfg["mapping"]["epochs"]
    print(f"[g_psi] {len(encodings)} cases, {epochs} epochs")

    for epoch in range(epochs):
        idxs = np.random.permutation(len(encodings))
        total = 0.0; nb = 0
        for b in range(0, len(idxs), bs):
            opt.zero_grad()
            batch = [encodings[i] for i in idxs[b:b + bs]]
            pred = model.g_psi(torch.stack([x[0] for x in batch]))
            loss = code_mse_loss(pred, torch.stack([x[1] for x in batch]))
            loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  epoch {epoch+1}: code MSE {total/max(nb,1):.6f}")

    torch.save({"g_psi_state": model.g_psi.state_dict(), "norm": norm},
               os.path.join(cfg["defaults"]["out_dir"], "infinity_full.pt"))
    print("Saved full model")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/infinity.yaml")
