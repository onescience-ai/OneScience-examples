"""GeoMPNN model package.

The GeoMPNN model source code is part of the AIRS library:
https://github.com/divelab/AIRS

Model architecture:
  - Surface encoder:    4 message-passing layers on airfoil surface
  - Surface-to-Volume:  4 message-passing layers to volume points
  - Hidden dimension:   64
  - Edge hidden:        64
  - RBF bases:          8
  - Coordinate encodings: hybrid polar-Cartesian + spherical harmonics

Available fields (each with a separate head):
  - ux / uy  → spherical harmonics head
  - p / nut  → inlet condition head (p uses log-pressure transform)

See geo_mpnn.py for usage examples.
"""
