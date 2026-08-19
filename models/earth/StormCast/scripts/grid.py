from __future__ import annotations

import numpy as np


def lambert_grid(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create the project's regional latitude/longitude grid in pure NumPy."""
    if image_size != (512, 640):
        raise ValueError("The regional grid must have shape (512, 640)")

    radius = 6371229.0
    standard_latitude = np.deg2rad(38.5)
    origin_latitude = np.deg2rad(38.5)
    central_longitude = np.deg2rad(-97.5)
    x = -2697520.1425219304 + 3000.0 * np.arange(1799, dtype=np.float64)
    y = -1587306.1525566636 + 3000.0 * np.arange(1059, dtype=np.float64)
    x = x[579:1219]
    y = y[273:785]
    xx, yy = np.meshgrid(x, y)

    n = np.sin(standard_latitude)
    f = np.cos(standard_latitude) * np.tan(np.pi / 4 + standard_latitude / 2) ** n / n
    rho0 = radius * f / np.tan(np.pi / 4 + origin_latitude / 2) ** n
    rho = np.hypot(xx, rho0 - yy)
    theta = np.arctan2(xx, rho0 - yy)
    latitude = 2 * np.arctan((radius * f / rho) ** (1 / n)) - np.pi / 2
    longitude = central_longitude + theta / n
    return np.rad2deg(latitude).astype(np.float32), np.mod(np.rad2deg(longitude), 360).astype(np.float32)
