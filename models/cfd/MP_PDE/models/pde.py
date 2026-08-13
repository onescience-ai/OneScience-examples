"""Paper-priority MP-PDE model for experiment E3."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn


class Swish(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return values * torch.sigmoid(values)


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
            Swish(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


def periodic_neighbor_indices(num_nodes: int, offsets: Iterable[int], device: torch.device | None = None) -> Tensor:
    """Return source indices [num_nodes, num_neighbors] for each target node."""
    offsets_tensor = torch.as_tensor(tuple(offsets), dtype=torch.long, device=device)
    if num_nodes < 7:
        raise ValueError(f"The six-neighbor periodic graph requires num_nodes>=7, found {num_nodes}")
    if offsets_tensor.numel() != 6 or offsets_tensor.unique().numel() != 6 or torch.any(offsets_tensor == 0):
        raise ValueError("MP-PDE E3 requires exactly six distinct non-zero neighbor offsets")
    target = torch.arange(num_nodes, dtype=torch.long, device=device)[:, None]
    return torch.remainder(target + offsets_tensor[None, :], num_nodes)


class MessagePassingLayer(nn.Module):
    """Equation (8)--(9) processor layer with sum aggregation."""

    def __init__(self, hidden_dim: int, history_dim: int, parameter_dim: int, affine_norm: bool):
        super().__init__()
        edge_dim = 2 * hidden_dim + history_dim + 1 + parameter_dim
        node_dim = 2 * hidden_dim + parameter_dim
        self.edge_mlp = TwoLayerMLP(edge_dim, hidden_dim, hidden_dim)
        self.node_mlp = TwoLayerMLP(node_dim, hidden_dim, hidden_dim)
        self.norm = nn.InstanceNorm1d(hidden_dim, affine=affine_norm, track_running_stats=False)

    def forward(
        self,
        hidden: Tensor,
        history: Tensor,
        x: Tensor,
        parameters: Tensor,
        neighbors: Tensor,
        domain_length: float,
    ) -> Tensor:
        batch, num_nodes, hidden_dim = hidden.shape
        num_neighbors = neighbors.shape[1]
        source_hidden = hidden[:, neighbors, :]
        target_hidden = hidden[:, :, None, :].expand(-1, -1, num_neighbors, -1)
        source_history = history[:, neighbors, :]
        history_difference = history[:, :, None, :] - source_history
        source_x = x[:, neighbors]
        displacement = x[:, :, None] - source_x
        displacement = torch.remainder(displacement + 0.5 * domain_length, domain_length) - 0.5 * domain_length
        theta_edges = parameters[:, None, None, :].expand(-1, num_nodes, num_neighbors, -1)
        edge_input = torch.cat(
            (target_hidden, source_hidden, history_difference, displacement[..., None], theta_edges), dim=-1
        )
        messages = self.edge_mlp(edge_input)
        aggregated = messages.sum(dim=2)
        theta_nodes = parameters[:, None, :].expand(-1, num_nodes, -1)
        node_update = self.node_mlp(torch.cat((hidden, aggregated, theta_nodes), dim=-1))
        return self.norm((hidden + node_update).transpose(1, 2)).transpose(1, 2)


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_window: int,
        middle_channels: int = 8,
        kernels: Sequence[int] = (16, 26),
        strides: Sequence[int] = (3, 1),
    ):
        super().__init__()
        if len(kernels) != 2 or len(strides) != 2:
            raise ValueError("The paper-priority decoder requires exactly two convolutions")
        length_after_first = (hidden_dim - int(kernels[0])) // int(strides[0]) + 1
        output_length = (length_after_first - int(kernels[1])) // int(strides[1]) + 1
        if output_length != time_window:
            raise ValueError(
                f"Decoder does not close hidden={hidden_dim} to K={time_window}: output length={output_length}"
            )
        self.network = nn.Sequential(
            nn.Conv1d(1, middle_channels, kernel_size=int(kernels[0]), stride=int(strides[0])),
            Swish(),
            nn.Conv1d(middle_channels, 1, kernel_size=int(kernels[1]), stride=int(strides[1])),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch, num_nodes, hidden_dim = hidden.shape
        decoded = self.network(hidden.reshape(batch * num_nodes, 1, hidden_dim))
        return decoded.reshape(batch, num_nodes, decoded.shape[-1])


class MPPDESolver(nn.Module):
    """Message-passing neural solver mapping K E3 states to the next K states."""

    def __init__(
        self,
        time_window: int = 25,
        hidden_dim: int = 164,
        message_passing_layers: int = 6,
        neighbor_offsets: Sequence[int] = (-3, -2, -1, 1, 2, 3),
        domain_length: float = 16.0,
        final_time: float = 4.0,
        parameter_maxima: Sequence[float] = (3.0, 0.4, 1.0),
        scale_coordinates: bool = True,
        scale_parameters: bool = True,
        instance_norm_affine: bool = False,
        decoder_middle_channels: int = 8,
        decoder_kernels: Sequence[int] = (16, 26),
        decoder_strides: Sequence[int] = (3, 1),
    ):
        super().__init__()
        if time_window <= 0 or hidden_dim <= 0 or message_passing_layers <= 0:
            raise ValueError("time_window, hidden_dim, and message_passing_layers must be positive")
        self.time_window = int(time_window)
        self.hidden_dim = int(hidden_dim)
        self.neighbor_offsets: Tuple[int, ...] = tuple(int(value) for value in neighbor_offsets)
        self.domain_length = float(domain_length)
        self.final_time = float(final_time)
        self.scale_coordinates = bool(scale_coordinates)
        self.scale_parameters = bool(scale_parameters)
        maxima = torch.tensor(tuple(float(value) for value in parameter_maxima), dtype=torch.float32)
        if maxima.shape != (3,) or torch.any(maxima <= 0.0):
            raise ValueError("parameter_maxima must contain three positive values")
        self.register_buffer("parameter_maxima", maxima, persistent=False)
        self.encoder = TwoLayerMLP(self.time_window + 1 + 1 + 3, self.hidden_dim, self.hidden_dim)
        self.processor = nn.ModuleList(
            MessagePassingLayer(self.hidden_dim, self.time_window, 3, instance_norm_affine)
            for _ in range(int(message_passing_layers))
        )
        self.decoder = TemporalDecoder(
            self.hidden_dim, self.time_window, decoder_middle_channels, decoder_kernels, decoder_strides
        )

    def _canonicalize_inputs(
        self, history: Tensor, x: Tensor, current_time: Tensor | float, parameters: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if history.ndim != 3:
            raise ValueError(f"history must have shape [B,N,K], found {tuple(history.shape)}")
        batch, num_nodes, window = history.shape
        if window != self.time_window:
            raise ValueError(f"Expected history K={self.time_window}, found {window}")
        if num_nodes < 7:
            raise ValueError(f"MP-PDE requires N>=7, found {num_nodes}")
        if parameters.shape != (batch, 3):
            raise ValueError(f"parameters must have shape [{batch},3], found {tuple(parameters.shape)}")
        if x.ndim == 1:
            if x.shape[0] != num_nodes:
                raise ValueError(f"x length {x.shape[0]} does not match N={num_nodes}")
            x = x[None, :].expand(batch, -1)
        elif x.shape != (batch, num_nodes):
            raise ValueError(f"x must have shape [N] or [B,N], found {tuple(x.shape)}")
        time = torch.as_tensor(current_time, dtype=history.dtype, device=history.device)
        if time.ndim == 0:
            time = time.expand(batch)
        elif time.shape == (batch, 1):
            time = time[:, 0]
        elif time.shape != (batch,):
            raise ValueError(f"current_time must be scalar or shape [B], found {tuple(time.shape)}")
        return history, x.to(history), time, parameters.to(history)

    def forward(
        self,
        history: Tensor,
        x: Tensor,
        current_time: Tensor | float,
        parameters: Tensor,
        dt: Tensor | float,
        *,
        return_derivative: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        history, x, current_time, parameters = self._canonicalize_inputs(history, x, current_time, parameters)
        batch, num_nodes, _ = history.shape
        scaled_x = x / self.domain_length if self.scale_coordinates else x
        scaled_time = current_time / self.final_time if self.scale_coordinates else current_time
        scaled_parameters = parameters / self.parameter_maxima.to(parameters) if self.scale_parameters else parameters
        time_feature = scaled_time[:, None, None].expand(-1, num_nodes, 1)
        parameter_features = scaled_parameters[:, None, :].expand(-1, num_nodes, -1)
        encoded = torch.cat((history, scaled_x[..., None], time_feature, parameter_features), dim=-1)
        hidden = self.encoder(encoded)
        neighbors = periodic_neighbor_indices(num_nodes, self.neighbor_offsets, history.device)
        for layer in self.processor:
            hidden = layer(hidden, history, x, scaled_parameters, neighbors, self.domain_length)
        derivative = self.decoder(hidden)
        step = torch.as_tensor(dt, dtype=history.dtype, device=history.device)
        if step.ndim == 0:
            step = step.expand(batch)
        elif step.shape != (batch,):
            raise ValueError(f"dt must be scalar or shape [B], found {tuple(step.shape)}")
        offsets = torch.arange(1, self.time_window + 1, dtype=history.dtype, device=history.device)
        delta_times = step[:, None, None] * offsets[None, None, :]
        prediction = history[:, :, -1:] + delta_times * derivative
        return (prediction, derivative) if return_derivative else prediction
