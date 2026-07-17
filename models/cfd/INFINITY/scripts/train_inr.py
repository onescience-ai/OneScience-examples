import os, sys, random, json, numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.config import load_config
from scripts.data.airfrans_dataset import AirfRANSDataPipe, build_normalizer, apply_normalizer
from scripts.data.sampling import sample_volume_points
from model.infinity import INFINITY, FUNCTIONS
from scripts.losses import inr_reconstruction_loss


def main(config_path):
    cfg = load_config(config_path)
    random.seed(cfg["defaults"]["seed"])
    np.random.seed(cfg["defaults"]["seed"])
    torch.manual_seed(cfg["defaults"]["seed"])

    data_root = cfg["defaults"]["data_root"]
    with open(cfg["defaults"]["manifest"]) as f:
        m = json.load(f)
    train_names = m[cfg["defaults"]["train_split"]]

    pipe = AirfRANSDataPipe(data_root, train_names)
    raw = [pipe[i] for i in range(len(pipe))]
    norm = build_normalizer(raw)
    cases = [apply_normalizer(c, norm) for c in raw]

    model = INFINITY(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.inrs.parameters(), lr=cfg["inr_train"]["outer_lr"],
                           weight_decay=cfg["optim"]["weight_decay"])
    n_pts = cfg["sampling"]["n_pts"]
    K = cfg["inr_train"]["K_inner"]
    inner_lr = cfg["inr_train"]["inner_lr"]
    ld = cfg["inr"]["latent_dim"]
    bs = cfg["inr_train"]["batch_size"]
    epochs = cfg["inr_train"]["epochs"]
    second_order = cfg["inr_train"].get("second_order", False)
    surface_bias = cfg["sampling"].get("surface_bias", 0.0)
    rng = np.random.RandomState(cfg["defaults"]["seed"])

    print(f"[INR] {len(cases)} cases, device={device}, K={K}, n_pts={n_pts}, "
          f"second_order={second_order}, surface_bias={surface_bias}")

    for epoch in range(epochs):
        random.shuffle(cases)
        total_loss = 0.0
        n_batches = 0
        for b in range(0, len(cases), bs):
            batch = cases[b:b + bs]
            opt.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for case in batch:
                vidx = sample_volume_points(case["x"], case["d"], n_pts,
                                            surface_bias, rng)
                vx = torch.tensor(case["x"][vidx], dtype=torch.float32, device=device)
                for u in FUNCTIONS:
                    inr = model.inrs[u]
                    if u == "d":
                        coords = torch.tensor(case["x"][:n_pts], dtype=torch.float32, device=device)
                        vals = torch.tensor(case["d"][:n_pts, 0], dtype=torch.float32, device=device)
                    elif u == "n":
                        coords = torch.tensor(case["surf_x"][:n_pts], dtype=torch.float32, device=device)
                        nv = np.concatenate([case["nx"], case["ny"]], 1)[:n_pts]
                        vals = torch.tensor(nv, dtype=torch.float32, device=device)
                    else:
                        col = {"vx": 0, "vy": 1, "p": 2, "nut": 3}[u]
                        coords = vx
                        vals = torch.tensor(case["y"][vidx, col], dtype=torch.float32, device=device)

                    z = torch.zeros(ld, device=device, requires_grad=True)
                    for _ in range(K):
                        phi = inr.hypernet(z.unsqueeze(0))[0]
                        pred = inr.forward(coords, phi)
                        l = ((pred - vals) ** 2).mean()
                        g = torch.autograd.grad(l, z, create_graph=second_order)[0]
                        if second_order:
                            z = z - inner_lr * g
                        else:
                            with torch.no_grad():
                                z = z - inner_lr * g
                            z.requires_grad_(True)

                    if not second_order:
                        z = z.requires_grad_(True)
                    phi = inr.hypernet(z.unsqueeze(0))[0]
                    pred = inr.forward(coords, phi)
                    outer_loss = ((pred - vals) ** 2).mean()
                    batch_loss = batch_loss + outer_loss

            batch_loss = batch_loss / (len(batch) * len(FUNCTIONS))
            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  epoch {epoch+1}: recon loss {avg_loss:.6f}")

    os.makedirs(cfg["defaults"]["out_dir"], exist_ok=True)
    torch.save({"inr_state": model.inrs.state_dict(), "norm": norm},
               os.path.join(cfg["defaults"]["out_dir"], "infinity_inr.pt"))
    print("Saved INR checkpoint")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/infinity.yaml")
