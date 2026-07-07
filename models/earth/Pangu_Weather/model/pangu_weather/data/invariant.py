from typing import Tuple

import numpy as np


def latlon_grid(
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((90, -90), (0, 360)),
    shape: Tuple[int, int] = (1440, 721),
):
    lat = np.linspace(*bounds[0], shape[0], dtype=np.float32)
    lon_wraparound = (bounds[1][0] % 360) == (bounds[1][1] % 360)
    if lon_wraparound:
        lon = np.linspace(*bounds[1], shape[1] + 1, dtype=np.float32)[:-1]
    else:
        lon = np.linspace(*bounds[1], shape[1], dtype=np.float32)
    return np.meshgrid(lat, lon, indexing="ij")
