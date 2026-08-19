"""Portable OneForecast model and official checkpoint compatibility helpers.

The parameter hierarchy mirrors the official model. Graph operations use
PyTorch index tensors instead of CUDA-only CuGraph kernels, making the model
usable on CPU, CUDA, and DCU PyTorch builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphData:
    """Minimal homogeneous or bipartite graph used by the portable kernels."""

    def __init__(self, src: Tensor, dst: Tensor, num_src: int, num_dst: int) -> None:
        self.src = src.to(torch.long)
        self.dst = dst.to(torch.long)
        self.num_src = num_src
        self.num_dst = num_dst

    def to(self, device: torch.device | str) -> "GraphData":
        self.src = self.src.to(device)
        self.dst = self.dst.to(device)
        return self


def _aggregate(values: Tensor, dst: Tensor, num_dst: int, reduction: str) -> Tensor:
    output = values.new_zeros((num_dst,) + values.shape[1:])
    index = dst.view((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
    output.scatter_add_(0, index, values)
    if reduction == "mean":
        counts = values.new_zeros(num_dst)
        counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=values.dtype))
        output = output / counts.clamp_min(1).view((-1,) + (1,) * (values.ndim - 1))
    elif reduction != "sum":
        raise ValueError(f"Unsupported aggregation: {reduction}")
    return output


def _edge_softmax(logits: Tensor, dst: Tensor, num_dst: int) -> Tensor:
    index = dst[:, None].expand_as(logits)
    maxima = logits.new_full((num_dst, logits.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - maxima[dst])
    denominator = logits.new_zeros((num_dst, logits.shape[1]))
    denominator.scatter_add_(0, index, exp)
    return exp / denominator[dst].clamp_min(torch.finfo(exp.dtype).tiny)


class MeshGraphMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512, hidden_dim: int = 512,
                 hidden_layers: int | None = 1, activation_fn: nn.Module | None = None,
                 norm_type: str | None = "LayerNorm", recompute_activation: bool = False) -> None:
        super().__init__()
        del recompute_activation
        activation_fn = activation_fn or nn.SiLU()
        if hidden_layers is None:
            self.model = nn.Identity()
            return
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation_fn]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if norm_type is not None:
            if norm_type != "LayerNorm":
                raise ValueError("The portable model supports LayerNorm only")
            layers.append(nn.LayerNorm(output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MeshGraphEdgeMLPSum(nn.Module):
    """Concat-trick edge MLP with the official parameter names and initialization."""

    def __init__(self, efeat_dim: int, src_dim: int, dst_dim: int,
                 output_dim: int = 512, hidden_dim: int = 512,
                 hidden_layers: int = 1, activation_fn: nn.Module | None = None,
                 norm_type: str | None = "LayerNorm", recompute_activation: bool = False) -> None:
        super().__init__()
        del recompute_activation
        activation_fn = activation_fn or nn.SiLU()
        initial = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim)
        weights = torch.split(initial.weight, [efeat_dim, src_dim, dst_dim], dim=1)
        self.lin_efeat = nn.Parameter(weights[0])
        self.lin_src = nn.Parameter(weights[1])
        self.lin_dst = nn.Parameter(weights[2])
        self.bias = initial.bias
        layers: list[nn.Module] = [activation_fn]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if norm_type is not None:
            if norm_type != "LayerNorm":
                raise ValueError("The portable model supports LayerNorm only")
            layers.append(nn.LayerNorm(output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, efeat: Tensor, nfeat: Tensor | tuple[Tensor, Tensor],
                graph: GraphData) -> Tensor:
        src_feat, dst_feat = (nfeat, nfeat) if isinstance(nfeat, Tensor) else nfeat
        hidden = F.linear(efeat, self.lin_efeat)
        hidden = hidden + F.linear(src_feat[graph.src], self.lin_src)
        hidden = hidden + F.linear(dst_feat[graph.dst], self.lin_dst, self.bias)
        return self.model(hidden)


class OneForecastEncoderEmbedder(nn.Module):
    def __init__(self, input_dim_grid_nodes: int = 69, input_dim_mesh_nodes: int = 3,
                 input_dim_edges: int = 4, output_dim: int = 512,
                 hidden_dim: int = 512, hidden_layers: int = 1) -> None:
        super().__init__()
        kwargs = dict(output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.grid_node_mlp = MeshGraphMLP(input_dim_grid_nodes, **kwargs)
        self.mesh_node_mlp = MeshGraphMLP(input_dim_mesh_nodes, **kwargs)
        self.mesh_edge_mlp = MeshGraphMLP(input_dim_edges, **kwargs)
        self.grid2mesh_edge_mlp = MeshGraphMLP(input_dim_edges, **kwargs)

    def forward(self, grid: Tensor, mesh: Tensor, g2m: Tensor,
                mesh_edges: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (self.grid_node_mlp(grid), self.mesh_node_mlp(mesh),
                self.grid2mesh_edge_mlp(g2m), self.mesh_edge_mlp(mesh_edges))


class OneForecastDecoderEmbedder(nn.Module):
    def __init__(self, input_dim_edges: int = 4, output_dim: int = 512,
                 hidden_dim: int = 512, hidden_layers: int = 1) -> None:
        super().__init__()
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim_edges, output_dim, hidden_dim, hidden_layers)

    def forward(self, edges: Tensor) -> Tensor:
        return self.mesh2grid_edge_mlp(edges)


class MeshGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, hidden_layers: int = 1,
                 aggregation: str = "sum") -> None:
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = MeshGraphEdgeMLPSum(hidden_dim, hidden_dim, hidden_dim,
                                            hidden_dim, hidden_dim, hidden_layers)
        self.src_node_mlp = MeshGraphMLP(hidden_dim, hidden_dim, hidden_dim, hidden_layers)
        self.dst_node_mlp = MeshGraphMLP(hidden_dim * 2, hidden_dim, hidden_dim, hidden_layers)

    def forward(self, edges: Tensor, grid: Tensor, mesh: Tensor,
                graph: GraphData) -> tuple[Tensor, Tensor]:
        edges = self.edge_mlp(edges, (grid, mesh), graph)
        aggregated = _aggregate(edges, graph.dst, graph.num_dst, self.aggregation)
        return grid + self.src_node_mlp(grid), mesh + self.dst_node_mlp(torch.cat((aggregated, mesh), -1))


class MeshGraphDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, hidden_layers: int = 1,
                 aggregation: str = "sum") -> None:
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = MeshGraphEdgeMLPSum(hidden_dim, hidden_dim, hidden_dim,
                                            hidden_dim, hidden_dim, hidden_layers)
        self.node_mlp = MeshGraphMLP(hidden_dim * 2, hidden_dim, hidden_dim, hidden_layers)

    def forward(self, edges: Tensor, grid: Tensor, mesh: Tensor, graph: GraphData) -> Tensor:
        edges = self.edge_mlp(edges, (mesh, grid), graph)
        aggregated = _aggregate(edges, graph.dst, graph.num_dst, self.aggregation)
        return grid + self.node_mlp(torch.cat((aggregated, grid), -1))


class MeshEdgeBlockMultiHeadGated(nn.Module):
    def __init__(self, hidden_dim: int = 512, hidden_layers: int = 1,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.edge_mlp = MeshGraphEdgeMLPSum(hidden_dim, hidden_dim, hidden_dim,
                                            hidden_dim, hidden_dim, hidden_layers)
        gating_hidden = max(16, hidden_dim // 8)
        self.gate_net = nn.Sequential(nn.Linear(hidden_dim * 3, gating_hidden), nn.SiLU(),
                                      nn.Linear(gating_hidden, 3 * num_heads), nn.Sigmoid())

    def forward(self, edges: Tensor, nodes: Tensor, graph: GraphData) -> tuple[Tensor, Tensor]:
        raw = torch.cat((edges, nodes[graph.src], nodes[graph.dst]), -1)
        gates = self.gate_net(raw).view(-1, self.num_heads, 3).mean(1)
        updated = self.edge_mlp(edges, nodes, graph)
        return edges + updated * gates.mean(-1, keepdim=True), nodes


class MeshNodeBlockMultiHeadAttn(nn.Module):
    def __init__(self, hidden_dim: int = 512, hidden_layers: int = 1,
                 aggregation: str = "sum", num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(hidden_dim * (num_heads + 1), hidden_dim,
                                     hidden_dim, hidden_layers)
        attention_hidden = max(16, hidden_dim // 8)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, attention_hidden), nn.SiLU(),
                                      nn.Linear(attention_hidden, num_heads))

    def forward(self, edges: Tensor, nodes: Tensor, graph: GraphData) -> tuple[Tensor, Tensor]:
        scores = _edge_softmax(self.attn_net(edges), graph.dst, graph.num_dst)
        messages = edges[:, None, :].expand(-1, self.num_heads, -1) * scores[:, :, None]
        aggregated = _aggregate(messages, graph.dst, graph.num_dst, self.aggregation).flatten(1)
        return edges, nodes + self.node_mlp(torch.cat((aggregated, nodes), -1))


class OneForecastProcessor(nn.Module):
    def __init__(self, processor_layers: int, hidden_dim: int = 512,
                 hidden_layers: int = 1, aggregation: str = "sum",
                 num_heads_edge: int = 4, num_heads_node: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlockMultiHeadGated(hidden_dim, hidden_layers, num_heads_edge))
            layers.append(MeshNodeBlockMultiHeadAttn(hidden_dim, hidden_layers, aggregation, num_heads_node))
        self.processor_layers = nn.ModuleList(layers)

    def forward(self, edges: Tensor, nodes: Tensor, graph: GraphData) -> tuple[Tensor, Tensor]:
        for layer in self.processor_layers:
            edges, nodes = layer(edges, nodes, graph)
        return edges, nodes


class TriangularMesh(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def _icosahedron() -> TriangularMesh:
    from scipy.spatial.transform import Rotation

    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for first in (1.0, -1.0):
        for second in (phi, -phi):
            vertices.extend(((first, second, 0.0), (0.0, first, second), (second, 0.0, first)))
    vertices = np.asarray(vertices, dtype=np.float32) / np.linalg.norm([1.0, phi])
    faces = np.asarray(((0,1,2),(0,6,1),(8,0,2),(8,4,0),(3,8,2),(3,2,7),(7,2,1),
                        (0,4,6),(4,11,6),(6,11,5),(1,5,7),(4,10,11),(4,8,10),(10,8,3),
                        (10,3,9),(11,10,9),(11,9,5),(5,9,7),(9,3,7),(1,6,5)), dtype=np.int32)
    angle = (np.pi - 2 * np.arcsin(phi / np.sqrt(3))) / 2
    vertices = vertices @ Rotation.from_euler("y", angle).as_matrix()
    return TriangularMesh(vertices.astype(np.float32), faces)


def _split_mesh(mesh: TriangularMesh) -> TriangularMesh:
    vertices = list(mesh.vertices)
    children: dict[tuple[int, int], int] = {}
    faces = []
    for a, b, c in mesh.faces:
        mids = []
        for pair in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(map(int, pair)))
            if key not in children:
                position = mesh.vertices[list(pair)].mean(0)
                position /= np.linalg.norm(position)
                children[key] = len(vertices)
                vertices.append(position)
            mids.append(children[key])
        ab, bc, ca = mids
        faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return TriangularMesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32))


def _mesh_hierarchy(level: int) -> list[TriangularMesh]:
    meshes = [_icosahedron()]
    for _ in range(level):
        meshes.append(_split_mesh(meshes[-1]))
    return meshes


def _faces_to_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2])),
            np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0])))


def _latlon_to_xyz(latlon: Tensor) -> Tensor:
    values = torch.deg2rad(latlon)
    lat, lon = values[:, 0], values[:, 1]
    return torch.stack((torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)), 1)


def _node_features(xyz: Tensor) -> Tensor:
    # The official implementation applies trigonometric functions to the
    # degree-valued xyz2latlon output; retain that behavior for parity.
    lat = torch.rad2deg(torch.asin(xyz[:, 2]))
    lon = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))
    return torch.stack((torch.cos(lat), torch.sin(lon), torch.cos(lon)), -1)


def _edge_features(src_pos: Tensor, dst_pos: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    source, target = src_pos[src], dst_pos[dst]
    lat = torch.asin(target[:, 2])
    lon = torch.atan2(target[:, 1], target[:, 0])
    cos_lon, sin_lon = torch.cos(-lon), torch.sin(-lon)
    source = torch.stack((cos_lon * source[:, 0] - sin_lon * source[:, 1],
                          sin_lon * source[:, 0] + cos_lon * source[:, 1], source[:, 2]), -1)
    target = torch.stack((cos_lon * target[:, 0] - sin_lon * target[:, 1],
                          sin_lon * target[:, 0] + cos_lon * target[:, 1], target[:, 2]), -1)
    cos_lat, sin_lat = torch.cos(lat), torch.sin(lat)
    source = torch.stack((cos_lat * source[:, 0] + sin_lat * source[:, 2], source[:, 1],
                          -sin_lat * source[:, 0] + cos_lat * source[:, 2]), -1)
    target = torch.stack((cos_lat * target[:, 0] + sin_lat * target[:, 2], target[:, 1],
                          -sin_lat * target[:, 0] + cos_lat * target[:, 2]), -1)
    displacement = source - target
    norm = torch.linalg.norm(displacement, dim=-1, keepdim=True)
    maximum = norm.max()
    return torch.cat((displacement / maximum, norm / maximum), -1)


def _local_refine(mesh: TriangularMesh, lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float) -> TriangularMesh:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    # Match the official xyz2latlon call, which assumes radius=1 for centroids.
    latitudes = np.rad2deg(np.arcsin(centroids[:, 2]))
    longitudes = np.rad2deg(np.arctan2(centroids[:, 1], centroids[:, 0]))
    selected = ((latitudes >= lat_min) & (latitudes <= lat_max)
                & (longitudes >= lon_min) & (longitudes <= lon_max))
    refined = _split_mesh(TriangularMesh(mesh.vertices, mesh.faces[selected]))
    combined_vertices = np.concatenate((refined.vertices, mesh.vertices), axis=0)
    combined_faces = np.concatenate((refined.faces, mesh.faces[~selected] + len(refined.vertices)), axis=0)
    rounded = np.round(combined_vertices, decimals=6)
    unique: dict[tuple[float, float, float], int] = {}
    remap = np.empty(len(rounded), dtype=np.int64)
    vertices = []
    for index, coordinates in enumerate(rounded):
        key = tuple(coordinates.tolist())
        if key not in unique:
            unique[key] = len(vertices)
            vertices.append(combined_vertices[index])
        remap[index] = unique[key]
    return TriangularMesh(np.asarray(vertices, dtype=np.float32), remap[combined_faces].astype(np.int32))


def _build_graphs(height: int, width: int, mesh_level: int) -> tuple[GraphData, GraphData, GraphData, Tensor, Tensor, Tensor, Tensor]:
    from scipy.spatial import cKDTree

    latitudes = torch.linspace(-90, 90, height + 1)[:-1]
    longitudes = torch.linspace(-180, 180, width + 1)[1:]
    latlon = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), -1).reshape(-1, 2)
    grid_xyz = _latlon_to_xyz(latlon)
    hierarchy = _mesh_hierarchy(mesh_level)
    finest = hierarchy[-1]
    refined = _local_refine(finest, 0.0, 30.0, 105.0, 160.0)
    refined = _local_refine(refined, 10.0, 30.0, -95.0, -35.0)
    mesh_vertices = refined.vertices
    mesh_faces = np.concatenate([mesh.faces for mesh in hierarchy] + [refined.faces], axis=0)
    mesh_src, mesh_dst = _faces_to_edges(mesh_faces)
    mesh_src = np.concatenate((mesh_src, mesh_dst))
    mesh_dst = np.concatenate((mesh_dst, mesh_src[:len(mesh_dst)]))
    pairs = np.unique(np.stack((mesh_src, mesh_dst), 1), axis=0)
    mesh_src_t = torch.from_numpy(pairs[:, 0])
    mesh_dst_t = torch.from_numpy(pairs[:, 1])
    mesh_xyz = torch.from_numpy(mesh_vertices)
    mesh_graph = GraphData(mesh_src_t, mesh_dst_t, len(mesh_vertices), len(mesh_vertices))

    finest_src, finest_dst = _faces_to_edges(finest.faces)
    max_edge = np.linalg.norm(finest.vertices[finest_src] - finest.vertices[finest_dst], axis=1).max()
    distances, neighbors = cKDTree(mesh_vertices).query(grid_xyz.numpy(), k=4)
    valid = distances <= 0.6 * max_edge
    g2m_src, neighbor_slot = np.nonzero(valid)
    g2m_dst = neighbors[g2m_src, neighbor_slot]
    g2m_graph = GraphData(torch.from_numpy(g2m_src), torch.from_numpy(g2m_dst), len(grid_xyz), len(mesh_vertices))

    centroids = mesh_vertices[mesh_faces].mean(axis=1)
    face_indices = cKDTree(centroids).query(grid_xyz.numpy(), k=1)[1]
    m2g_src = mesh_faces[face_indices].reshape(-1)
    m2g_dst = np.repeat(np.arange(len(grid_xyz)), 3)
    m2g_graph = GraphData(torch.from_numpy(m2g_src), torch.from_numpy(m2g_dst), len(mesh_vertices), len(grid_xyz))
    mesh_nodes = _node_features(mesh_xyz)
    mesh_edges = _edge_features(mesh_xyz, mesh_xyz, mesh_graph.src, mesh_graph.dst)
    g2m_edges = _edge_features(grid_xyz, mesh_xyz, g2m_graph.src, g2m_graph.dst)
    m2g_edges = _edge_features(mesh_xyz, grid_xyz, m2g_graph.src, m2g_graph.dst)
    return mesh_graph, g2m_graph, m2g_graph, mesh_nodes, mesh_edges, g2m_edges, m2g_edges


class OneForecast(nn.Module):
    """Official OneForecast message-passing architecture with portable graph kernels."""

    def __init__(self, input_res: tuple[int, int] = (120, 240), input_dim_grid_nodes: int = 69,
                 output_dim_grid_nodes: int = 69, mesh_level: int = 5,
                 processor_layers: int = 16, hidden_layers: int = 1,
                 hidden_dim: int = 512, aggregation: str = "sum",
                 num_heads_edge: int = 4, num_heads_node: int = 4,
                 build_graph: bool = True) -> None:
        super().__init__()
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        self.register_buffer("device_buffer", torch.empty(0))
        self.input_res = tuple(input_res)
        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.mesh_level = mesh_level
        self.encoder_embedder = OneForecastEncoderEmbedder(
            input_dim_grid_nodes, 3, 4, hidden_dim, hidden_dim, hidden_layers)
        self.decoder_embedder = OneForecastDecoderEmbedder(4, hidden_dim, hidden_dim, hidden_layers)
        self.encoder = MeshGraphEncoder(hidden_dim, hidden_layers, aggregation)
        self.processor_encoder = OneForecastProcessor(
            1, hidden_dim, hidden_layers, aggregation, num_heads_edge, num_heads_node)
        self.processor = OneForecastProcessor(
            processor_layers - 2, hidden_dim, hidden_layers, aggregation, num_heads_edge, num_heads_node)
        self.processor_decoder = OneForecastProcessor(
            1, hidden_dim, hidden_layers, aggregation, num_heads_edge, num_heads_node)
        self.decoder = MeshGraphDecoder(hidden_dim, hidden_layers, aggregation)
        self.finale = MeshGraphMLP(hidden_dim, output_dim_grid_nodes, hidden_dim, hidden_layers, norm_type=None)
        self._graph_ready = False
        if build_graph:
            self.build_graph()

    def build_graph(self) -> None:
        values = _build_graphs(*self.input_res, self.mesh_level)
        self.mesh_graph, self.g2m_graph, self.m2g_graph = values[:3]
        for name, value in zip(("mesh_ndata", "mesh_edata", "g2m_edata", "m2g_edata"), values[3:]):
            self.register_buffer(name, value, persistent=False)
        self._graph_ready = True

    def forward(self, grid_nfeat: Tensor) -> Tensor:
        if not self._graph_ready:
            raise RuntimeError("Graph construction was disabled for this model instance")
        if grid_nfeat.shape != (1, self.input_dim_grid_nodes, *self.input_res):
            raise ValueError(f"Expected input shape (1, {self.input_dim_grid_nodes}, {self.input_res[0]}, {self.input_res[1]}), got {tuple(grid_nfeat.shape)}")
        grid = grid_nfeat[0].reshape(self.input_dim_grid_nodes, -1).T
        grid, mesh, g2m, mesh_edges = self.encoder_embedder(
            grid, self.mesh_ndata, self.g2m_edata, self.mesh_edata)
        grid, mesh = self.encoder(g2m, grid, mesh, self.g2m_graph)
        mesh_edges, mesh = self.processor_encoder(mesh_edges, mesh, self.mesh_graph)
        mesh_edges, mesh = self.processor(mesh_edges, mesh, self.mesh_graph)
        _, mesh = self.processor_decoder(mesh_edges, mesh, self.mesh_graph)
        grid = self.decoder(self.decoder_embedder(self.m2g_edata), grid, mesh, self.m2g_graph)
        output = self.finale(grid).T.reshape(self.output_dim_grid_nodes, *self.input_res)
        return output.unsqueeze(0)

    def to(self, *args: Any, **kwargs: Any) -> "OneForecast":
        super().to(*args, **kwargs)
        if self._graph_ready:
            device = self.device_buffer.device
            self.mesh_graph.to(device)
            self.g2m_graph.to(device)
            self.m2g_graph.to(device)
        return self


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint_path: str
    checkpoint_keys: int
    model_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


def _install_scalarfloat_safe_global() -> type[float]:
    """Allow the known ruamel ScalarFloat metadata type without importing ruamel."""
    module_name = "ruamel.yaml.scalarfloat"
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, "ScalarFloat"):
        scalar_float = module.ScalarFloat
    else:
        ruamel = sys.modules.setdefault("ruamel", types.ModuleType("ruamel"))
        yaml_module = sys.modules.setdefault("ruamel.yaml", types.ModuleType("ruamel.yaml"))
        module = types.ModuleType(module_name)
        scalar_float = type("ScalarFloat", (float,), {})
        scalar_float.__module__ = module_name
        module.ScalarFloat = scalar_float
        yaml_module.scalarfloat = module
        ruamel.yaml = yaml_module
        sys.modules[module_name] = module
    torch.serialization.add_safe_globals([scalar_float])
    return scalar_float


def read_official_checkpoint(path: str | Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    _install_scalarfloat_safe_global()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{path} does not contain an official model_state")
    state = checkpoint["model_state"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint model_state must be a mapping")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state", "optimizer_state_dict"}}
    return cleaned, metadata


def _compare_checkpoint_state(model: nn.Module, checkpoint_state: dict[str, Tensor],
                              path: str | Path) -> CheckpointReport:
    model_state = model.state_dict()
    missing = tuple(sorted(set(model_state) - set(checkpoint_state)))
    unexpected = tuple(sorted(set(checkpoint_state) - set(model_state)))
    mismatches = tuple(sorted(
        f"{key}: checkpoint={tuple(checkpoint_state[key].shape)} model={tuple(model_state[key].shape)}"
        for key in set(model_state) & set(checkpoint_state)
        if model_state[key].shape != checkpoint_state[key].shape
    ))
    return CheckpointReport(str(Path(path).expanduser().resolve()), len(checkpoint_state),
                            len(model_state), missing, unexpected, mismatches)


def check_checkpoint_compatibility(model: nn.Module, path: str | Path) -> CheckpointReport:
    checkpoint_state, _ = read_official_checkpoint(path)
    return _compare_checkpoint_state(model, checkpoint_state, path)


def load_official_checkpoint(model: nn.Module, path: str | Path, strict: bool = True) -> CheckpointReport:
    state, _ = read_official_checkpoint(path)
    report = _compare_checkpoint_state(model, state, path)
    if strict and not report.compatible:
        raise RuntimeError(f"Official checkpoint is incompatible: {report}")
    compatible = {key: value for key, value in state.items()
                  if key in model.state_dict() and value.shape == model.state_dict()[key].shape}
    model.load_state_dict(compatible, strict=strict)
    return report


def build_model(config: dict[str, Any], build_graph: bool = True) -> OneForecast:
    settings = config["model"]
    model = OneForecast(
        input_res=(settings["grid_height"], settings["grid_width"]),
        input_dim_grid_nodes=settings["input_channels"],
        output_dim_grid_nodes=settings["output_channels"],
        mesh_level=settings.get("mesh_level", 5),
        processor_layers=settings.get("processor_layers", 16),
        hidden_layers=settings.get("hidden_layers", 1),
        hidden_dim=settings.get("hidden_dim", 512),
        num_heads_edge=settings.get("num_heads_edge", 4),
        num_heads_node=settings.get("num_heads_node", 4),
        build_graph=build_graph,
    )
    initialization = settings.get("weight_init", "scratch")
    if initialization == "official":
        load_official_checkpoint(model, settings["checkpoint_path"])
    elif initialization != "scratch":
        raise ValueError("model.weight_init must be 'scratch' or 'official'")
    return model


__all__ = ["CheckpointReport", "OneForecast", "build_model", "check_checkpoint_compatibility",
           "load_official_checkpoint", "read_official_checkpoint"]
