"""Small VTK XML reader for AirfRANS `.vtu` and `.vtp` files.

The reader supports inline zlib-compressed binary arrays, which is the format
used by the local Transolver-Airfoil-Design dataset. It avoids a hard pyvista
dependency for training, while evaluation can still use pyvista when available.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np


_DTYPES = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt32": np.uint32,
}


@dataclass
class VTKMesh:
    path: Path
    piece_attributes: dict[str, str]
    points: np.ndarray | None
    point_data: dict[str, np.ndarray]
    cell_data: dict[str, np.ndarray]
    cells: dict[str, np.ndarray]


def _compact_text(text: str | None) -> str:
    return "".join((text or "").split())


def _base64_len(raw_len: int) -> int:
    return int(4 * math.ceil(raw_len / 3))


def _decode_binary_payload(payload: str, dtype_name: str) -> np.ndarray:
    if dtype_name not in _DTYPES:
        raise ValueError(f"Unsupported VTK dtype: {dtype_name}")
    if len(payload) < 16:
        raise ValueError("Binary VTK payload is too short to contain a compressed header")

    first_header = base64.b64decode(payload[:16])
    num_blocks, _block_size, _last_block_size = struct.unpack("<III", first_header)
    header_bytes = (3 + num_blocks) * 4
    header_b64 = payload[: _base64_len(header_bytes)]
    header = struct.unpack("<" + "I" * (3 + num_blocks), base64.b64decode(header_b64))
    compressed_sizes = header[3:]
    compressed_blob = base64.b64decode(payload[_base64_len(header_bytes) :])

    offset = 0
    chunks: list[bytes] = []
    for size in compressed_sizes:
        block = compressed_blob[offset : offset + size]
        chunks.append(zlib.decompress(block))
        offset += size

    raw = b"".join(chunks)
    return np.frombuffer(raw, dtype=_DTYPES[dtype_name]).copy()


def _read_data_array(element: ET.Element) -> tuple[str, np.ndarray] | None:
    name = element.attrib.get("Name")
    dtype_name = element.attrib.get("type")
    fmt = element.attrib.get("format")
    if not name or not dtype_name:
        return None
    if fmt != "binary":
        raise ValueError(f"Only binary VTK arrays are supported; got {fmt!r} for {name}")

    payload = _compact_text(element.text)
    if not payload:
        return None
    array = _decode_binary_payload(payload, dtype_name)
    components = int(element.attrib.get("NumberOfComponents", "1"))
    if components > 1:
        if array.size % components != 0:
            raise ValueError(f"Array {name} size {array.size} is not divisible by {components}")
        array = array.reshape(-1, components)
    return name, array


def _read_section(section: ET.Element | None, wanted: set[str] | None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if section is None:
        return result
    for data_array in section.findall("DataArray"):
        parsed = _read_data_array(data_array)
        if parsed is None:
            continue
        name, array = parsed
        if wanted is None or name in wanted:
            result[name] = array
    return result


def read_vtk_xml(
    path: str | Path,
    *,
    point_arrays: list[str] | None = None,
    cell_arrays: list[str] | None = None,
    read_points: bool = True,
    read_cells: bool = False,
) -> VTKMesh:
    """Read selected arrays from a `.vtp` or `.vtu` file."""

    vtk_path = Path(path)
    root = ET.parse(vtk_path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"No VTK Piece element found in {vtk_path}")

    points = None
    if read_points:
        point_section = piece.find("Points")
        point_arrays_dict = _read_section(point_section, {"Points"})
        points = point_arrays_dict.get("Points")

    cells: dict[str, np.ndarray] = {}
    if read_cells:
        cells = _read_section(piece.find("Cells"), None)

    point_data = _read_section(piece.find("PointData"), set(point_arrays) if point_arrays else None)
    cell_data = _read_section(piece.find("CellData"), set(cell_arrays) if cell_arrays else None)
    return VTKMesh(
        path=vtk_path,
        piece_attributes=dict(piece.attrib),
        points=points,
        point_data=point_data,
        cell_data=cell_data,
        cells=cells,
    )
