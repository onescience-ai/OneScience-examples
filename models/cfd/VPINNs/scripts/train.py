from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.vpinn import (  # noqa: E402
    VPINN,
    d_test_function_jacobi,
    gauss_lobatto_jacobi_weights,
    test_function_jacobi,
)


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
CASES = ("1d", "2d")
OMEGA_1D = 8.0 * np.pi
STEEPNESS_1D = 80.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "root" not in config:
        raise ValueError(f"config must contain a 'root' mapping: {path}")
    return config["root"]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_finite(loss: torch.Tensor, case: str, step: int) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{case} loss became non-finite at step {step}")


def positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def save_checkpoint(weight_dir: Path, filename: str, payload: dict) -> Path:
    weight_dir.mkdir(parents=True, exist_ok=True)
    output_path = weight_dir / filename
    torch.save(payload, output_path)
    print(f"Saved checkpoint: {output_path}")
    return output_path


def exact_poisson1d(x: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(OMEGA_1D * x) + np.tanh(STEEPNESS_1D * x)


def source_poisson1d(x: np.ndarray) -> np.ndarray:
    oscillatory = -0.1 * OMEGA_1D**2 * np.sin(OMEGA_1D * x)
    transition = (
        -2.0
        * STEEPNESS_1D**2
        * np.tanh(STEEPNESS_1D * x)
        / np.cosh(STEEPNESS_1D * x) ** 2
    )
    return -(oscillatory + transition)


def build_rhs_1d(
    element_grid: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
    n_test: int,
) -> np.ndarray:
    test_values = np.stack(
        [
            test_function_jacobi(index + 1, quadrature_points)
            for index in range(n_test)
        ]
    )
    rhs = np.empty((element_grid.size - 1, n_test), dtype=float)
    for element in range(element_grid.size - 1):
        left, right = element_grid[element : element + 2]
        jacobian = (right - left) / 2.0
        physical_points = left + jacobian * (quadrature_points + 1.0)
        weighted_source = quadrature_weights * source_poisson1d(physical_points)
        rhs[element] = jacobian * (test_values @ weighted_source)
    return rhs


def train_poisson1d(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    weight_dir: Path,
) -> None:
    n_element = positive(int(config["n_element"]), "n_element")
    n_test = positive(int(config["n_test"]), "n_test")
    n_quad = positive(int(config["n_quad"]), "n_quad")
    epochs = int(config["epochs"])
    lbfgs_iters = int(config["lbfgs_iters"])
    if epochs < 0 or lbfgs_iters < 0:
        raise ValueError("epochs and lbfgs_iters must be non-negative")
    layers = [int(value) for value in config["layers"]]
    if layers[0] != 1 or layers[-1] != 1:
        raise ValueError("1D Poisson layers must start and end with width 1")

    element_grid = np.linspace(-1.0, 1.0, n_element + 1)
    quadrature_points, quadrature_weights = gauss_lobatto_jacobi_weights(n_quad)
    rhs = build_rhs_1d(element_grid, quadrature_points, quadrature_weights, n_test)
    test_derivatives = np.stack(
        [
            d_test_function_jacobi(index + 1, quadrature_points)[0]
            for index in range(n_test)
        ]
    )

    model = VPINN(layers, dtype=dtype).to(device=device, dtype=dtype)
    quadrature_tensor = torch.as_tensor(quadrature_points, dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(quadrature_weights, dtype=dtype, device=device)
    derivative_tensor = torch.as_tensor(test_derivatives, dtype=dtype, device=device)
    rhs_tensor = torch.as_tensor(rhs, dtype=dtype, device=device)
    boundary = torch.tensor([[-1.0], [1.0]], dtype=dtype, device=device)
    boundary_values = torch.as_tensor(
        exact_poisson1d(np.array([-1.0, 1.0]))[:, None], dtype=dtype, device=device
    )
    test_points = torch.linspace(-1.0, 1.0, 2001, dtype=dtype, device=device).unsqueeze(-1)
    exact_values = torch.as_tensor(
        exact_poisson1d(test_points.cpu().numpy().reshape(-1))[:, None],
        dtype=dtype,
        device=device,
    )

    def variational_loss() -> torch.Tensor:
        loss = torch.zeros((), dtype=dtype, device=device)
        for element in range(n_element):
            left, right = element_grid[element : element + 2]
            jacobian = (right - left) / 2.0
            physical_points = (
                left + jacobian * (quadrature_tensor + 1.0)
            ).unsqueeze(-1)
            physical_points.requires_grad_(True)
            prediction = model(physical_points)
            prediction_x = torch.autograd.grad(
                prediction,
                physical_points,
                torch.ones_like(prediction),
                create_graph=True,
            )[0]
            weak_prediction = derivative_tensor @ (
                weight_tensor * prediction_x.reshape(-1)
            )
            loss = loss + torch.mean((weak_prediction - rhs_tensor[element]).square())
        boundary_loss = torch.mean((model(boundary) - boundary_values).square())
        return loss + boundary_loss

    def relative_l2() -> float:
        with torch.no_grad():
            return (
                torch.linalg.vector_norm(model(test_points) - exact_values)
                / torch.linalg.vector_norm(exact_values)
            ).item()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    log_every = int(config["log_every"])
    started = time.time()
    print(
        f"[1D] device={device} elements={n_element} n_test={n_test} "
        f"n_quad={n_quad} epochs={epochs} lbfgs_iters={lbfgs_iters}"
    )
    for step in range(1, epochs + 1):
        loss = variational_loss()
        ensure_finite(loss, "1D Poisson", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == epochs:
            print(
                f"[1D] step={step:5d} loss={loss.item():.3e} "
                f"relative_l2={relative_l2():.3e}"
            )

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            max_eval=max(1, 2 * lbfgs_iters),
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss = variational_loss()
            ensure_finite(closure_loss, "1D Poisson L-BFGS", 0)
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

    error = relative_l2()
    print(f"[1D] finished in {time.time() - started:.1f}s, relative L2={error:.6e}")
    save_checkpoint(
        weight_dir,
        "hpvpinn_poisson1d.pt",
        {
            "case": "1d",
            "architecture": "vpinn",
            "model_state": model.state_dict(),
            "layers": layers,
            "n_element": n_element,
            "n_test": n_test,
            "n_quad": n_quad,
        },
    )


def exact_poisson2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (0.1 * np.sin(2.0 * np.pi * x) + np.tanh(10.0 * x)) * np.sin(
        2.0 * np.pi * y
    )


def source_poisson2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_component = (
        -0.4 * np.pi**2 * np.sin(2.0 * np.pi * x)
        - 200.0 * np.tanh(10.0 * x) / np.cosh(10.0 * x) ** 2
    ) * np.sin(2.0 * np.pi * y)
    y_component = (
        0.1 * np.sin(2.0 * np.pi * x) + np.tanh(10.0 * x)
    ) * (-4.0 * np.pi**2 * np.sin(2.0 * np.pi * y))
    return x_component + y_component


def build_rhs_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
    n_test: int,
) -> np.ndarray:
    test_values = np.stack(
        [
            test_function_jacobi(index + 1, quadrature_points)
            for index in range(n_test)
        ]
    )
    n_elements_x = grid_x.size - 1
    n_elements_y = grid_y.size - 1
    rhs = np.empty((n_elements_x, n_elements_y, n_test, n_test), dtype=float)
    for element_x in range(n_elements_x):
        left_x, right_x = grid_x[element_x : element_x + 2]
        jacobian_x = (right_x - left_x) / 2.0
        physical_x = left_x + jacobian_x * (quadrature_points + 1.0)
        for element_y in range(n_elements_y):
            left_y, right_y = grid_y[element_y : element_y + 2]
            jacobian_y = (right_y - left_y) / 2.0
            physical_y = left_y + jacobian_y * (quadrature_points + 1.0)
            mesh_x, mesh_y = np.meshgrid(physical_x, physical_y, indexing="ij")
            weighted_source = (
                quadrature_weights[:, None]
                * source_poisson2d(mesh_x, mesh_y)
                * quadrature_weights[None, :]
            )
            rhs[element_x, element_y] = (
                jacobian_x
                * jacobian_y
                * (test_values @ weighted_source @ test_values.T)
            )
    return rhs


def sample_boundary_2d(n_bound: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    values = []
    random_x = 2.0 * np.random.rand(n_bound, 1) - 1.0
    for boundary_y in (1.0, -1.0):
        y = np.full_like(random_x, boundary_y)
        points.append(np.hstack((random_x, y)))
        values.append(exact_poisson2d(random_x, y))
    random_y = 2.0 * np.random.rand(n_bound, 1) - 1.0
    for boundary_x in (1.0, -1.0):
        x = np.full_like(random_y, boundary_x)
        points.append(np.hstack((x, random_y)))
        values.append(exact_poisson2d(x, random_y))
    return np.vstack(points), np.vstack(values)


def train_poisson2d(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    weight_dir: Path,
) -> None:
    n_el_x = positive(int(config["n_el_x"]), "n_el_x")
    n_el_y = positive(int(config["n_el_y"]), "n_el_y")
    n_test = positive(int(config["n_test"]), "n_test")
    n_quad = positive(int(config["n_quad"]), "n_quad")
    n_bound = positive(int(config["n_bound"]), "n_bound")
    epochs = int(config["epochs"])
    lbfgs_iters = int(config["lbfgs_iters"])
    if epochs < 0 or lbfgs_iters < 0:
        raise ValueError("epochs and lbfgs_iters must be non-negative")
    layers = [int(value) for value in config["layers"]]
    if layers[0] != 2 or layers[-1] != 1:
        raise ValueError("2D Poisson layers must start with width 2 and end with width 1")

    grid_x = np.linspace(-1.0, 1.0, n_el_x + 1)
    grid_y = np.linspace(-1.0, 1.0, n_el_y + 1)
    quadrature_points, quadrature_weights = gauss_lobatto_jacobi_weights(n_quad)
    rhs = build_rhs_2d(
        grid_x, grid_y, quadrature_points, quadrature_weights, n_test
    )
    test_values = np.stack(
        [
            test_function_jacobi(index + 1, quadrature_points)
            for index in range(n_test)
        ]
    )
    test_derivatives = np.stack(
        [
            d_test_function_jacobi(index + 1, quadrature_points)[0]
            for index in range(n_test)
        ]
    )
    boundary_points, boundary_values = sample_boundary_2d(n_bound)

    model = VPINN(layers, dtype=dtype).to(device=device, dtype=dtype)
    quadrature_tensor = torch.as_tensor(quadrature_points, dtype=dtype, device=device)
    weight_tensor = torch.as_tensor(quadrature_weights, dtype=dtype, device=device)
    test_tensor = torch.as_tensor(test_values, dtype=dtype, device=device)
    derivative_tensor = torch.as_tensor(test_derivatives, dtype=dtype, device=device)
    rhs_tensor = torch.as_tensor(rhs, dtype=dtype, device=device)
    boundary_tensor = torch.as_tensor(boundary_points, dtype=dtype, device=device)
    boundary_value_tensor = torch.as_tensor(boundary_values, dtype=dtype, device=device)

    evaluation_axis = torch.linspace(-1.0, 1.0, 100, dtype=dtype, device=device)
    evaluation_x, evaluation_y = torch.meshgrid(
        evaluation_axis, evaluation_axis, indexing="ij"
    )
    evaluation_points = torch.stack(
        (evaluation_x.reshape(-1), evaluation_y.reshape(-1)), dim=-1
    )
    exact_values = torch.as_tensor(
        exact_poisson2d(
            evaluation_points[:, 0].cpu().numpy(),
            evaluation_points[:, 1].cpu().numpy(),
        )[:, None],
        dtype=dtype,
        device=device,
    )

    def variational_loss() -> torch.Tensor:
        loss = torch.zeros((), dtype=dtype, device=device)
        for element_x in range(n_el_x):
            left_x, right_x = grid_x[element_x : element_x + 2]
            jacobian_x = (right_x - left_x) / 2.0
            physical_x = left_x + jacobian_x * (quadrature_tensor + 1.0)
            for element_y in range(n_el_y):
                left_y, right_y = grid_y[element_y : element_y + 2]
                jacobian_y = (right_y - left_y) / 2.0
                physical_y = left_y + jacobian_y * (quadrature_tensor + 1.0)
                mesh_x, mesh_y = torch.meshgrid(physical_x, physical_y, indexing="ij")
                coordinates = torch.stack(
                    (mesh_x.reshape(-1), mesh_y.reshape(-1)), dim=-1
                )
                coordinates.requires_grad_(True)
                prediction = model(coordinates)
                prediction_gradient = torch.autograd.grad(
                    prediction,
                    coordinates,
                    torch.ones_like(prediction),
                    create_graph=True,
                )[0]
                prediction_x = prediction_gradient[:, 0].reshape(n_quad, n_quad)
                prediction_y = prediction_gradient[:, 1].reshape(n_quad, n_quad)
                weighted_x = (
                    weight_tensor[:, None] * prediction_x * weight_tensor[None, :]
                )
                weighted_y = (
                    weight_tensor[:, None] * prediction_y * weight_tensor[None, :]
                )
                weak_prediction = -jacobian_y * (
                    derivative_tensor @ weighted_x @ test_tensor.T
                ) - jacobian_x * (test_tensor @ weighted_y @ derivative_tensor.T)
                loss = loss + torch.mean(
                    (weak_prediction - rhs_tensor[element_x, element_y]).square()
                )
        boundary_loss = torch.mean(
            (model(boundary_tensor) - boundary_value_tensor).square()
        )
        return loss + float(config["boundary_weight"]) * boundary_loss

    def relative_l2() -> float:
        with torch.no_grad():
            return (
                torch.linalg.vector_norm(model(evaluation_points) - exact_values)
                / torch.linalg.vector_norm(exact_values)
            ).item()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    log_every = int(config["log_every"])
    started = time.time()
    print(
        f"[2D] device={device} elements={n_el_x}x{n_el_y} n_test={n_test} "
        f"n_quad={n_quad} epochs={epochs} lbfgs_iters={lbfgs_iters}"
    )
    for step in range(1, epochs + 1):
        loss = variational_loss()
        ensure_finite(loss, "2D Poisson", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == epochs:
            print(
                f"[2D] step={step:5d} loss={loss.item():.3e} "
                f"relative_l2={relative_l2():.3e}"
            )

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            max_eval=max(1, 2 * lbfgs_iters),
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss = variational_loss()
            ensure_finite(closure_loss, "2D Poisson L-BFGS", 0)
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

    error = relative_l2()
    print(f"[2D] finished in {time.time() - started:.1f}s, relative L2={error:.6e}")
    save_checkpoint(
        weight_dir,
        "hpvpinn_poisson2d.pt",
        {
            "case": "2d",
            "architecture": "vpinn",
            "model_state": model.state_dict(),
            "layers": layers,
            "n_el_x": n_el_x,
            "n_el_y": n_el_y,
            "n_test": n_test,
            "n_quad": n_quad,
        },
    )


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(common["weight_dir"])
    seed_everything(int(common["seed"]))

    print(f"Config: {config_path}")
    selected_case = str(common["case"]).lower()
    if selected_case not in (*CASES, "all"):
        raise ValueError("common.case must be one of: 1d, 2d, all")
    selected_cases = CASES if selected_case == "all" else (selected_case,)
    if "1d" in selected_cases:
        train_poisson1d(config["poisson1d"], device, dtype, weight_dir)
    if "2d" in selected_cases:
        train_poisson2d(config["poisson2d"], device, dtype, weight_dir)


if __name__ == "__main__":
    main()
