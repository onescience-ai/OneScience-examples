import numpy as np


def sample_volume_points(x, d, n_pts, surface_bias=0.0, rng=None):
    """Sample n_pts volume node indices.

    surface_bias in [0,1): fraction of points drawn with distance-weighted
    oversampling of near-surface nodes (small |implicit_distance|), where the
    pressure stagnation peak lives. The remaining fraction is drawn uniformly.
    """
    N = x.shape[0]
    rng = rng or np.random
    if N <= n_pts:
        return np.arange(N)

    if surface_bias <= 0.0:
        return rng.choice(N, n_pts, replace=False)

    n_bias = int(round(n_pts * surface_bias))
    n_unif = n_pts - n_bias

    abs_d = np.abs(np.asarray(d).reshape(-1))
    scale = max(float(np.percentile(abs_d, 20)), 1e-6)
    w = 1.0 / (1.0 + (abs_d / scale) ** 2)
    w = w / w.sum()

    bias_idx = rng.choice(N, n_bias, replace=False, p=w)
    if n_unif > 0:
        unif_idx = rng.choice(N, n_unif, replace=False)
        return np.concatenate([bias_idx, unif_idx])
    return bias_idx
