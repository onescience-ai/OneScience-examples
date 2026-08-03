"""
TensorNEAT Prototype: Jena Weather Forecasting
================================================
Evolve neural network topology + weights using NEAT (NeuroEvolution of
Augmenting Topologies) to predict temperature from the Jena Climate dataset.

Approach:
- Select 3 key features, subsample to 1h intervals
- Sliding window of past W hours → predict T(degC) H hours ahead
- Use FuncFit base class (supervised regression) with NEAT evolution
- RecurrentGenome with activate_time for implicit temporal processing

Results (50 generations, 300 population, CPU):
- Test RMSE: 3.518 °C
- Test MAE:  2.797 °C
- 13.8% improvement over persistence baseline
"""

from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import RecurrentGenome, BiasNode
from tensorneat.genome.operations.mutation.recurrent import RecurrentMutation
from tensorneat.problem.func_fit import FuncFit
from tensorneat.common import ACT, AGG, State


# =============================================================================
# 1. DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_jena(
    csv_path="jena_climate_2009_2016.csv",
    subsample_step=6,       # every 6th row → 1h intervals (from 10min)
    window_size=24,         # 24 hours of history
    forecast_horizon=6,     # predict 6 hours ahead
    max_train_samples=2000, # limit for CPU-based NEAT (keep tractable)
    max_test_samples=500,
    target_col="T (degC)",
):
    """Load Jena Climate data and create sliding-window regression dataset."""
    
    print("Loading Jena Climate dataset...")
    df = pd.read_csv(csv_path)
    
    # --- Clean erroneous wind speed values (-9999 sentinel) ---
    df["wv (m/s)"] = df["wv (m/s)"].replace(-9999.0, 0.0)
    df["max. wv (m/s)"] = df["max. wv (m/s)"].replace(-9999.0, 0.0)
    
    # --- Select key features (minimal set for prototype) ---
    # Keep only 3 most informative + non-redundant features
    selected_features = [
        "T (degC)",       # temperature (our target + most predictive)
        "p (mbar)",       # atmospheric pressure
        "rh (%)",         # relative humidity
    ]
    
    df = df[selected_features]
    
    # --- Subsample to 1-hour intervals ---
    df = df.iloc[::subsample_step].reset_index(drop=True)
    print(f"After subsampling: {len(df)} rows, {len(selected_features)} features")
    
    # --- Normalize using training set statistics ---
    # Use first 70% for train stats
    n_total = len(df)
    n_train_end = int(n_total * 0.7)
    
    train_mean = df.iloc[:n_train_end].mean()
    train_std = df.iloc[:n_train_end].std()
    train_std = train_std.replace(0, 1)  # avoid division by zero
    
    df_norm = (df - train_mean) / train_std
    
    data = df_norm.values.astype(np.float32)
    target_idx = selected_features.index(target_col)
    
    print(f"Feature stats (pre-norm train):")
    for i, feat in enumerate(selected_features):
        print(f"  {feat}: mean={train_mean.iloc[i]:.2f}, std={train_std.iloc[i]:.2f}")
    
    # --- Create sliding window dataset ---
    # Input: flatten [window_size × num_features] → [window_size * num_features]
    # Target: T(degC) at t + forecast_horizon
    num_features = len(selected_features)
    
    X_list = []
    y_list = []
    
    for i in range(window_size, len(data) - forecast_horizon):
        window = data[i - window_size : i]  # (window_size, num_features)
        target = data[i + forecast_horizon, target_idx]  # scalar
        X_list.append(window.flatten())
        y_list.append(target)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    
    # --- Split into train/test ---
    n_samples = len(X)
    split_idx = int(n_samples * 0.7)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # --- Subsample for tractability on CPU ---
    if len(X_train) > max_train_samples:
        # Take evenly spaced samples to preserve temporal coverage
        indices = np.linspace(0, len(X_train) - 1, max_train_samples, dtype=int)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    if len(X_test) > max_test_samples:
        indices = np.linspace(0, len(X_test) - 1, max_test_samples, dtype=int)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    print(f"\nDataset created:")
    print(f"  Window size: {window_size}h, Forecast horizon: {forecast_horizon}h")
    print(f"  Input dim: {X_train.shape[1]} ({window_size} × {num_features} features)")
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Target: {target_col} (normalized)")
    print(f"  y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    return X_train, y_train, X_test, y_test, train_mean, train_std, target_idx


# =============================================================================
# 2. CUSTOM PROBLEM CLASS
# =============================================================================

class JenaWeatherProblem(FuncFit):
    """
    TensorNEAT Problem for Jena weather forecasting.
    
    Inherits from FuncFit which provides:
    - evaluate(): vmaps act_func over inputs, computes error vs targets
    - show(): prints input/target/prediction comparison
    
    We just need to provide inputs, targets, and their shapes.
    """
    jitable = True
    
    def __init__(self, X, y, error_method="mse"):
        self._inputs = jnp.array(X, dtype=jnp.float32)
        self._targets = jnp.array(y, dtype=jnp.float32)
        super().__init__(error_method=error_method)
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def input_shape(self):
        return self._inputs.shape  # (N_samples, input_dim)
    
    @property
    def output_shape(self):
        return self._targets.shape  # (N_samples, 1)


# =============================================================================
# 3. EVALUATION ON TEST SET
# =============================================================================

def evaluate_on_test(pipeline, state, best, X_test, y_test, train_mean, train_std, target_idx):
    """Evaluate best evolved genome on the test set."""
    
    test_inputs = jnp.array(X_test, dtype=jnp.float32)
    test_targets = jnp.array(y_test, dtype=jnp.float32)
    
    # Get transformed genome for forward pass
    transformed = pipeline.algorithm.transform(state, best)
    
    # Predict on all test samples
    predictions = vmap(pipeline.algorithm.forward, in_axes=(None, None, 0))(
        state, transformed, test_inputs
    )
    
    predictions = jax.device_get(predictions)
    test_targets = jax.device_get(test_targets)
    
    # Compute metrics in normalized space
    mse_norm = np.mean((predictions - test_targets) ** 2)
    mae_norm = np.mean(np.abs(predictions - test_targets))
    
    # Denormalize for interpretable metrics
    t_mean = train_mean.iloc[target_idx]
    t_std = train_std.iloc[target_idx]
    
    pred_denorm = predictions * t_std + t_mean
    target_denorm = test_targets * t_std + t_mean
    
    mse_real = np.mean((pred_denorm - target_denorm) ** 2)
    mae_real = np.mean(np.abs(pred_denorm - target_denorm))
    rmse_real = np.sqrt(mse_real)
    
    # Baseline: predict last known temperature (persistence model)
    # Last T value in each window is at position: (window_size-1) * num_features + target_idx
    inferred_window = X_test.shape[1] // len(train_mean)
    last_t_idx = (inferred_window - 1) * len(train_mean) + target_idx
    baseline_pred = X_test[:, last_t_idx:last_t_idx+1]
    baseline_mse = np.mean((baseline_pred - y_test) ** 2)
    baseline_mae = np.mean(np.abs(baseline_pred - y_test))
    baseline_rmse = np.sqrt(baseline_mse)
    
    baseline_pred_denorm = baseline_pred * t_std + t_mean
    baseline_mse_real = np.mean((baseline_pred_denorm - target_denorm) ** 2)
    baseline_mae_real = np.mean(np.abs(baseline_pred_denorm - target_denorm))
    baseline_rmse_real = np.sqrt(baseline_mse_real)
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nNEAT Evolved Network:")
    print(f"  MSE  (normalized): {mse_norm:.6f}")
    print(f"  MAE  (normalized): {mae_norm:.6f}")
    print(f"  RMSE (°C):         {rmse_real:.3f}")
    print(f"  MAE  (°C):         {mae_real:.3f}")
    print(f"\nBaseline (persistence — last known T):")
    print(f"  MSE  (normalized): {baseline_mse:.6f}")
    print(f"  MAE  (normalized): {baseline_mae:.6f}")
    print(f"  RMSE (°C):         {baseline_rmse_real:.3f}")
    print(f"  MAE  (°C):         {baseline_mae_real:.3f}")
    
    improvement = (baseline_mae_real - mae_real) / baseline_mae_real * 100
    print(f"\nImprovement over baseline: {improvement:+.1f}% MAE reduction")
    
    # Show some example predictions
    print(f"\nSample predictions (first 10):")
    print(f"  {'Predicted °C':>14}  {'Actual °C':>10}  {'Error °C':>9}")
    for i in range(min(10, len(pred_denorm))):
        err = pred_denorm[i, 0] - target_denorm[i, 0]
        print(f"  {pred_denorm[i, 0]:14.2f}  {target_denorm[i, 0]:10.2f}  {err:+9.2f}")
    
    return {
        "mse_norm": float(mse_norm),
        "mae_norm": float(mae_norm),
        "rmse_celsius": float(rmse_real),
        "mae_celsius": float(mae_real),
        "baseline_rmse_celsius": float(baseline_rmse_real),
        "baseline_mae_celsius": float(baseline_mae_real),
    }


# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("TensorNEAT: Jena Weather Forecasting Prototype")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print()
    
    # --- Load & preprocess data ---
    # V1 prototype: use 3 key features + short window to keep tractable on CPU
    # NEAT tensor size scales with max_nodes^2, so fewer inputs = much faster
    WINDOW_SIZE = 3          # 3 hours of history 
    FORECAST_HORIZON = 6     # predict 6 hours ahead
    MAX_TRAIN = 500          # keep small for CPU
    MAX_TEST = 200
    
    X_train, y_train, X_test, y_test, train_mean, train_std, target_idx = \
        load_and_preprocess_jena(
            csv_path=str(Path(__file__).resolve().parent / "data" / "jena_climate_2009_2016.csv"),
            subsample_step=6,
            window_size=WINDOW_SIZE,
            forecast_horizon=FORECAST_HORIZON,
            max_train_samples=MAX_TRAIN,
            max_test_samples=MAX_TEST,
        )
    
    NUM_INPUTS = X_train.shape[1]   # window_size * 3 features = 9
    NUM_OUTPUTS = 1                 # predict T(degC)
    
    # --- Define problem ---
    problem = JenaWeatherProblem(X_train, y_train, error_method="mse")
    
    # --- Configure NEAT evolution ---
    # max_nodes must be >= num_inputs + num_outputs + room for hidden nodes
    # With 3 features × 3 window = 9 inputs, this is very tractable
    POP_SIZE = 300
    GENERATIONS = 50
    MAX_HIDDEN = 10  # allow up to 10 hidden nodes to be evolved
    MAX_NODES = NUM_INPUTS + NUM_OUTPUTS + MAX_HIDDEN  # 9 + 1 + 10 = 20
    MAX_CONNS = 80   # connection budget
    
    print(f"\nNEAT Configuration:")
    print(f"  Population: {POP_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Network inputs: {NUM_INPUTS}, outputs: {NUM_OUTPUTS}")
    print(f"  Max nodes: {MAX_NODES} (inputs+outputs+{MAX_HIDDEN} hidden)")
    print(f"  Max connections: {MAX_CONNS}")
    print(f"  Genome type: RecurrentGenome (activate_time=5)")
    print()
    
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=POP_SIZE,
            species_size=10,
            survival_threshold=0.1,
            compatibility_threshold=2.0,
            max_stagnation=15,
            genome=RecurrentGenome(
                num_inputs=NUM_INPUTS,
                num_outputs=NUM_OUTPUTS,
                max_nodes=MAX_NODES,
                max_conns=MAX_CONNS,
                init_hidden_layers=(),  # start from minimal topology
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.sigmoid, ACT.relu, ACT.identity],
                    aggregation_options=[AGG.sum],
                    activation_replace_rate=0.1,
                ),
                mutation=RecurrentMutation(
                    conn_add=0.3,
                    conn_delete=0.05,
                    node_add=0.1,
                    node_delete=0.01,
                ),
                output_transform=ACT.identity,  # regression → no squashing
                activate_time=5,                # recurrent iterations per forward
            ),
        ),
        problem=problem,
        seed=42,
        generation_limit=GENERATIONS,
        fitness_target=-0.001,  # -MSE; stop if MSE < 0.001
        pop_batch_size=50,      # evaluate in batches of 50 for memory
    )
    
    # --- Run evolution ---
    print("Setting up pipeline...")
    state = pipeline.setup()
    
    print("\nStarting evolution...")
    state, best = pipeline.auto_run(state)
    
    # --- Evaluate on test set ---
    metrics = evaluate_on_test(
        pipeline, state, best,
        X_test, y_test,
        train_mean, train_std, target_idx
    )
    
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best training fitness: {pipeline.best_fitness:.6f}")
    print(f"Test RMSE: {metrics['rmse_celsius']:.3f} °C")
    print(f"Test MAE:  {metrics['mae_celsius']:.3f} °C")
