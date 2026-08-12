#!/usr/bin/env python3
"""使用官方单步 EDM 去噪目标训练 GenCast。"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import xarray

# Mesh adjacency construction triggers one-time scipy CSR restructure warning.
warnings.filterwarnings("ignore", message="Changing the sparsity structure")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.common import configure_jax, load_config, load_stats, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf/config.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--parallel-mode", choices=("single", "pmap"))
    parser.add_argument("--num-devices", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _adam_init(params):
    import jax
    import jax.numpy as jnp

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"count": jnp.asarray(0, dtype=jnp.int32), "mu": zeros, "nu": zeros}


def _adam_update(params, grads, state, learning_rate, beta1, beta2, eps):
    import jax
    import jax.numpy as jnp

    count = state["count"] + 1
    mu = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state["mu"], grads,
    )
    nu = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state["nu"], grads,
    )
    mu_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**count), mu)
    nu_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**count), nu)
    params = jax.tree_util.tree_map(
        lambda value, first, second: value - learning_rate * first / (jnp.sqrt(second) + eps),
        params, mu_hat, nu_hat,
    )
    return params, {"count": count, "mu": mu, "nu": nu}


def _replicate(tree, devices):
    import jax

    return jax.device_put_replicated(tree, devices)


def _unreplicate(tree):
    import jax

    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _device_batch(batch, device_count):
    """Add a leading device dimension to each GenCast xarray input."""
    result = []
    for value in batch:
        if not isinstance(value, xarray.Dataset):
            raise TypeError("GenCast batches must contain xarray.Dataset values")
        value = value.transpose("batch", ...)
        if "batch" not in value.dims:
            value = value.expand_dims("batch")
        if value.sizes["batch"] % device_count:
            raise ValueError("Batch size must be divisible by the device count")
        local_batch = value.sizes["batch"] // device_count
        shards = [
            value.isel(batch=slice(index * local_batch, (index + 1) * local_batch))
            for index in range(device_count)
        ]
        result.append(xarray.concat(shards, dim="device"))
    return tuple(result)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parallel = config.setdefault("parallel", {})
    if args.parallel_mode is not None:
        parallel["mode"] = args.parallel_mode
    if args.num_devices is not None:
        parallel["num_devices"] = args.num_devices
    if args.global_batch_size is not None:
        parallel["global_batch_size"] = args.global_batch_size
    if args.checkpoint is not None:
        config["checkpoint"]["trainer"] = args.checkpoint
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    configure_jax(config["runtime"].get("platform", "auto"))

    import jax
    import jax.numpy as jnp

    from model.gencast import GenCastModel, parameter_count
    from model.common import (
        load_trainer_checkpoint, save_trainer_checkpoint,
        validate_checkpoint_config,
    )
    from model.data_loader import GenCastERA5Dataset, batch_iterator

    mode = str(parallel.get("mode", "single")).lower()
    if mode not in ("single", "pmap"):
        raise ValueError("parallel.mode must be 'single' or 'pmap'")
    devices = list(jax.local_devices())
    requested_devices = int(parallel.get("num_devices", 1))
    if requested_devices < 1:
        raise ValueError("parallel.num_devices must be positive")
    if mode == "pmap":
        if requested_devices > len(devices):
            raise ValueError(
                f"Requested {requested_devices} devices, only {len(devices)} available"
            )
        devices = devices[:requested_devices]
    else:
        requested_devices = 1
        devices = devices[:1]
    global_batch_size = int(parallel.get("global_batch_size", requested_devices))
    if global_batch_size < 1 or global_batch_size % requested_devices:
        raise ValueError("global_batch_size must be divisible by the device count")
    stats = load_stats(config["data"]["stats_dir"])
    model = GenCastModel.from_config_and_stats(config, stats)
    dataset = GenCastERA5Dataset(
        resolve_path(config["data"]["data_dir"]),
        list(config["data"]["train_years"]),
        static_dir=resolve_path(config["data"]["static_dir"]),
        prediction_steps=1,
        stride=int(config["data"].get("train_stride", 1)),
        precipitation_interval_hours=int(
            config["data"]["precipitation_interval_hours"]
        ),
    )
    first_batch = dataset[0]
    seed = int(config["training"]["seed"])
    start_step = 0
    resume = args.resume or config["checkpoint"].get("resume")
    if resume:
        params, state, optimizer_state, start_step, saved_config = \
            load_trainer_checkpoint(resume)
        validate_checkpoint_config(config, saved_config)
    else:
        params, state = model.init(
            jax.random.fold_in(jax.random.PRNGKey(seed), -1), *first_batch
        )
        optimizer_state = _adam_init(params)

    learning_rate = float(config["training"]["learning_rate"])
    beta1, beta2 = (float(value) for value in config["training"]["betas"])
    epsilon = float(config["training"].get("epsilon", 1e-8))

    def train_step(params, state, optimizer_state, rng, inputs, targets, forcings):
        def objective(current_params, current_state):
            (loss, diagnostics), next_state = model.loss(
                current_params, current_state, rng, inputs, targets, forcings
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(params, state)
        finite = jnp.logical_and(
            jnp.isfinite(loss),
            jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)])),
        )
        new_params, new_optimizer_state = _adam_update(
            params, grads, optimizer_state, learning_rate, beta1, beta2, epsilon
        )
        params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old), new_params, params
        )
        next_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old), next_state, state
        )
        new_optimizer_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old),
            new_optimizer_state,
            optimizer_state,
        )
        return params, next_state, new_optimizer_state, loss, diagnostics, finite

    if mode == "pmap":
        axis_name = str(parallel.get("axis_name", "devices"))

        def parallel_train_step(
            params, state, optimizer_state, rng, inputs, targets, forcings
        ):
            rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name))

            def objective(current_params, current_state):
                (loss, diagnostics), next_state = model.loss(
                    current_params, current_state, rng, inputs, targets, forcings
                )
                return loss, (diagnostics, next_state)

            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params, state)
            grads = jax.lax.pmean(grads, axis_name)
            loss = jax.lax.pmean(loss, axis_name)
            diagnostics = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, axis_name), diagnostics
            )
            next_state = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, axis_name), next_state
            )
            finite = jnp.logical_and(
                jnp.isfinite(loss),
                jnp.all(jnp.asarray([
                    jnp.all(jnp.isfinite(x))
                    for x in jax.tree_util.tree_leaves(grads)
                ])),
            )
            finite = jax.lax.pmin(finite, axis_name)
            new_params, new_optimizer_state = _adam_update(
                params, grads, optimizer_state, learning_rate, beta1, beta2, epsilon
            )
            params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), new_params, params
            )
            next_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), next_state, state
            )
            new_optimizer_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old),
                new_optimizer_state,
                optimizer_state,
            )
            return params, next_state, new_optimizer_state, loss, diagnostics, finite

        from model.graphcast import xarray_jax

        train_step = xarray_jax.pmap(
            parallel_train_step, dim="device", axis_name=axis_name, devices=devices
        )
    else:
        train_step = jax.jit(train_step)
    max_steps = int(args.max_steps or config["training"]["max_steps"])
    save_interval = int(config["training"].get("save_interval", max_steps))
    checkpoint_path = config["checkpoint"]["trainer"]
    print(f"Training samples: {len(dataset)}; parameters: {parameter_count(params):,}")
    if mode == "pmap":
        params = _replicate(params, devices)
        state = _replicate(state, devices)
        optimizer_state = _replicate(optimizer_state, devices)
        print(
            f"Parallel mode: pmap; devices: {requested_devices}; "
            f"global batch: {global_batch_size}"
        )

    step = start_step
    batches_per_epoch = len(dataset) // global_batch_size
    if batches_per_epoch < 1:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than global_batch_size "
            f"{global_batch_size}"
        )
    while step < max_steps:
        epoch = step // batches_per_epoch
        offset = step % batches_per_epoch
        epoch_batches = batch_iterator(
            dataset,
            shuffle=True,
            seed=seed + epoch,
            batch_size=global_batch_size,
        )
        for batch in itertools.islice(epoch_batches, offset, None):
            if step >= max_steps:
                break
            step_rng = jax.random.fold_in(jax.random.PRNGKey(seed), step)
            if mode == "pmap":
                batch = _device_batch(batch, requested_devices)
                step_rng = jax.numpy.broadcast_to(
                    step_rng, (requested_devices, *step_rng.shape)
                )
                params, state, optimizer_state, loss, _, finite = train_step(
                    params, state, optimizer_state, step_rng, *batch
                )
                loss, finite = loss[0], finite[0]
            else:
                params, state, optimizer_state, loss, _, finite = train_step(
                    params, state, optimizer_state, step_rng, *batch
                )
            step += 1
            print(f"step={step} loss={float(loss):.8f} finite={bool(finite)}")
            if not bool(finite):
                raise FloatingPointError(f"Non-finite GenCast loss at step {step}")
            if step % save_interval == 0 or step == max_steps:
                checkpoint_trees = (params, state, optimizer_state)
                if mode == "pmap":
                    checkpoint_trees = tuple(map(_unreplicate, checkpoint_trees))
                save_trainer_checkpoint(
                    checkpoint_path,
                    params=checkpoint_trees[0],
                    state=checkpoint_trees[1],
                    optimizer_state=checkpoint_trees[2],
                    step=step,
                    config=config,
                )
                print(f"Saved checkpoint to {resolve_path(checkpoint_path)}")


if __name__ == "__main__":
    main()
