import logging
from functools import partial
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt

from common import load_config
from onescience.datapipes.cfd import DeepMindLagrangianDatapipe, graph_update
from onescience.distributed.manager import DistributedManager
from onescience.launch.utils import load_checkpoint

matplotlib.use("Agg")


TYPE_TO_COLOR = {
    0: "green",
    3: "black",
    5: "blue",
    6: "gold",
    7: "magenta",
}


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("lagrangian_mgn.inference")


def resolve_device(requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MGNRollout:
    def __init__(self, cfg, device, logger):
        if int(cfg.test.batch_size) != 1:
            raise ValueError(f"Only batch size 1 is supported, got {cfg.test.batch_size}")

        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.frame_skip = int(cfg.inference.frame_skip)

        self.datapipe = DeepMindLagrangianDatapipe(cfg, distributed=False)
        self.dataloader = self.datapipe.test_dataloader()
        self.dataset = self.datapipe.test_dataset

        self.dim = self.dataset.dim
        self.radius = self.dataset.radius
        self.dt = self.dataset.dt
        self.bounds = self.dataset.bounds
        self.num_history = self.dataset.num_history
        self.num_node_types = self.dataset.num_node_types

        self.model = hydra.utils.instantiate(cfg.model)
        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args)
        self.model = self.model.to(device)

        epoch = load_checkpoint(cfg.resume_dir, models=self.model, device=device)
        if epoch == 0:
            logger.warning("No checkpoint found in %s; running with random weights.", cfg.resume_dir)
        self.model.eval()

    def compute_boundary_feature(self, position):
        dist = torch.cat(
            [position - self.bounds[0], self.bounds[1] - position], dim=-1
        )
        feat = torch.exp(-(dist**2) / self.radius**2)
        feat[dist > self.radius] = 0
        return feat

    def boundary_clamp(self, position):
        min_bound = self.bounds[0] + 1e-3
        max_bound = self.bounds[1] - 1e-3
        return torch.clamp(position, min=min_bound, max=max_bound)

    def pack_inputs(self, position, vel_history, node_type):
        bound_feat = self.compute_boundary_feature(position)
        vel_hist = vel_history.permute(1, 0, 2).flatten(1)
        return torch.cat((position, vel_hist, bound_feat, node_type), dim=-1)

    @torch.inference_mode()
    def predict(self):
        pred_pos, gt_pos, node_type = [], [], None

        for graph in self.dataloader:
            graph = graph.to(self.device)

            if graph.ndata["t"][0].item() == 0:
                if pred_pos:
                    yield torch.stack(pred_pos), torch.stack(gt_pos), node_type

                pred_pos, gt_pos = [], []
                position, vel_history, node_type = self.dataset.unpack_inputs(graph)
                position = position.clone()
                vel_history = vel_history.clone()
                pred_pos.append(position)
                gt_pos.append(position)

            graph.ndata["x"] = self.pack_inputs(position, vel_history, node_type)
            graph.ndata["pos"] = position
            graph_update(graph, self.radius)

            acceleration = self.model(graph.ndata["x"], graph.edata["x"], graph)
            next_pos, next_vel = self.dataset.time_integrator(
                position=position,
                velocity=vel_history[-1],
                acceleration=acceleration,
                dt=self.dt,
                denormalize=True,
            )
            next_pos = self.boundary_clamp(next_pos)
            next_vel_norm = (
                next_vel - self.dataset.vel_mean.to(self.device)
            ) / self.dataset.vel_std.to(self.device)

            vel_history = torch.cat((vel_history[1:], next_vel_norm.unsqueeze(0)), dim=0)
            position = next_pos

            pred_pos.append(position)
            gt_next_pos, _, _ = self.dataset.unpack_targets(graph)
            gt_pos.append(gt_next_pos)

        if pred_pos:
            yield torch.stack(pred_pos), torch.stack(gt_pos), node_type


def init_animation(subplot_kw=None):
    return plt.subplots(1, 2, figsize=(12, 6), subplot_kw=subplot_kw)


def plot_particles_2d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.set_aspect("equal")
    ax.scatter(position[:, 0], position[:, 1], c=node_color, s=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_title(title)


def plot_particles_3d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.scatter(position[:, 2], position[:, 0], position[:, 1], c=node_color, s=10)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])
    ax.set_title(title)


def animate_frame(num, plotter, fig, ax1, ax2, pred, gt, node_color, bounds, frame_skip):
    num *= frame_skip
    if num >= len(pred):
        return
    plotter(ax1, "Prediction", pred[num], node_color, bounds)
    plotter(ax2, "Ground Truth", gt[num], node_color, bounds)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)


def save_animation(pred, gt, node_color, bounds, frame_skip, output_path, dim):
    if dim == 2:
        fig, (ax1, ax2) = init_animation()
        plotter = plot_particles_2d
    elif dim == 3:
        fig, (ax1, ax2) = init_animation(subplot_kw={"projection": "3d"})
        plotter = plot_particles_3d
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    ani = animation.FuncAnimation(
        fig,
        partial(
            animate_frame,
            plotter=plotter,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            pred=pred,
            gt=gt,
            node_color=node_color,
            bounds=bounds,
            frame_skip=frame_skip,
        ),
        frames=max((pred.shape[0] - 1) // frame_skip, 1),
        interval=100,
    )
    ani.save(output_path, writer="pillow", fps=10)
    plt.close(fig)


def main():
    cfg = load_config()
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    device = resolve_device(str(cfg.test.device))
    logger.info("Starting rollout on %s", device)

    rollout = MGNRollout(cfg, device, logger)
    mse_list = []

    for index, (pred_pos, gt_pos, node_type) in enumerate(rollout.predict()):
        pred = pred_pos.cpu().numpy()
        gt = gt_pos.cpu().numpy()
        node_type_np = node_type.cpu().numpy()
        mse = float(np.mean((pred - gt) ** 2))
        mse_list.append(mse)

        np.savez(output_dir / f"sequence_{index}.npz", prediction=pred, target=gt)
        logger.info("Sequence %s MSE %.4e", index, mse)

        if cfg.inference.save_animations:
            node_color = [
                TYPE_TO_COLOR.get(node_id, "gray")
                for node_id in np.argmax(node_type_np, axis=1)
            ]
            save_animation(
                pred,
                gt,
                node_color,
                rollout.bounds,
                rollout.frame_skip,
                output_dir / f"animation_{index}.gif",
                int(cfg.dim),
            )

        if index + 1 >= int(cfg.inference.max_sequences):
            break

    np.savez(output_dir / "rollout_metrics.npz", mse=np.asarray(mse_list, dtype=np.float32))
    logger.info("Average MSE %.4e", float(np.mean(mse_list)) if mse_list else float("nan"))
    manager.cleanup()


if __name__ == "__main__":
    main()
