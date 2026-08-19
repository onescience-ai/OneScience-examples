from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import torch
from torch import nn


def _install_nvtx_fallback() -> None:
    try:
        __import__("nvtx")
        return
    except ModuleNotFoundError:
        pass

    class _Annotate(contextlib.ContextDecorator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _Annotate:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    module = types.ModuleType("nvtx")
    module.annotate = _Annotate
    sys.modules["nvtx"] = module


_install_nvtx_fallback()

try:
    __import__("onescience.models.module")
except ModuleNotFoundError:
    from onescience.modules import module as _onescience_module

    sys.modules["onescience.models.module"] = _onescience_module

from onescience.models.diffusion.song_unet import SongUNet  # noqa: E402


class _StormCastSongUNet(SongUNet):
    """OneScience SongUNet with StormCast's optional learned spatial embedding."""

    def __init__(
        self,
        *args: Any,
        additive_pos_embed: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.additive_pos_embed = additive_pos_embed
        if additive_pos_embed:
            model_channels = self.enc[next(iter(self.enc))].out_channels
            self.spatial_emb = nn.Parameter(
                torch.empty(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            nn.init.trunc_normal_(self.spatial_emb, std=0.02)
            first_layer = self.enc[next(iter(self.enc))]
            first_layer.register_forward_hook(self._add_spatial_embedding)

    def _add_spatial_embedding(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.shape[-2:] != self.spatial_emb.shape[-2:]:
            raise ValueError(
                "Input grid must match img_resolution when additive_pos_embed is enabled"
            )
        return output + self.spatial_emb.to(dtype=output.dtype)


class StormCastRegressionUNet(nn.Module):
    """Deterministic regression wrapper."""

    def __init__(
        self,
        img_resolution: int | list[int] | tuple[int, int],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
        model_type: str = "SongUNet",
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        if model_type != "SongUNet":
            raise ValueError("Regression requires model_type='SongUNet'")
        self.register_buffer("device_buffer", torch.empty(0))
        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = _StormCastSongUNet(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(self, x: torch.Tensor, force_fp32: bool = False) -> torch.Tensor:
        _validate_image(x, "x", self.img_in_channels)
        dtype = _model_dtype(x, self.use_fp16, force_fp32)
        output = self.model(
            x.to(dtype),
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device),
            class_labels=None,
        )
        _validate_output_dtype(output, dtype)
        return output.to(torch.float32)


class StormCastEDMPrecond(nn.Module):
    """EDM preconditioner for conditional residual diffusion."""

    def __init__(
        self,
        img_resolution: int | list[int] | tuple[int, int],
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
        model_type: str = "SongUNet",
        img_in_channels: int | None = None,
        img_out_channels: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        if model_type != "SongUNet":
            raise ValueError("Diffusion requires model_type='SongUNet'")
        if label_dim != 0:
            raise ValueError("Diffusion does not use class labels")
        self.register_buffer("device_buffer", torch.empty(0))
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.img_in_channels = img_channels if img_in_channels is None else img_in_channels
        self.img_out_channels = img_channels if img_out_channels is None else img_out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = _StormCastSongUNet(
            img_resolution=img_resolution,
            in_channels=self.img_in_channels,
            out_channels=self.img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor | None = None,
        force_fp32: bool = False,
    ) -> torch.Tensor:
        _validate_image(x, "x", self.img_out_channels)
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(
            -1, 1, 1, 1
        )
        if sigma.shape[0] not in (1, x.shape[0]):
            raise ValueError("sigma must contain one value or one value per batch item")

        c_skip = self.sigma_data**2 / (sigma.square() + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma.square() + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma.square()).sqrt()
        c_noise = sigma.log() / 4
        model_input = c_in * x.to(torch.float32)
        if condition is not None:
            _validate_condition(condition, x)
            model_input = torch.cat((model_input, condition.to(torch.float32)), dim=1)
        if model_input.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Diffusion model expects {self.img_in_channels} total channels, "
                f"got {model_input.shape[1]}"
            )

        dtype = _model_dtype(x, self.use_fp16, force_fp32)
        output = self.model(
            model_input.to(dtype), c_noise.flatten(), class_labels=None
        )
        _validate_output_dtype(output, dtype)
        return c_skip * x.to(torch.float32) + c_out * output.to(torch.float32)

    @staticmethod
    def round_sigma(sigma: float | list[float] | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


class StormCast(nn.Module):
    """Compose regression and conditional residual diffusion stages."""

    def __init__(
        self,
        regression: StormCastRegressionUNet,
        diffusion: StormCastEDMPrecond,
    ) -> None:
        super().__init__()
        self.regression = regression
        self.diffusion = diffusion

    def regression_condition(
        self,
        state: torch.Tensor,
        background: torch.Tensor,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        invariant = _expand_invariant(invariant, state.shape[0])
        return torch.cat((state, background, invariant), dim=1)

    def diffusion_condition(
        self,
        state: torch.Tensor,
        regression: torch.Tensor,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        invariant = _expand_invariant(invariant, state.shape[0])
        return torch.cat((state, regression, invariant), dim=1)

    def predict_regression(
        self,
        state: torch.Tensor,
        background: torch.Tensor,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        return self.regression(self.regression_condition(state, background, invariant))

    def denoise_residual(
        self,
        noisy_residual: torch.Tensor,
        sigma: torch.Tensor,
        state: torch.Tensor,
        regression: torch.Tensor,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.diffusion_condition(state, regression, invariant)
        return self.diffusion(noisy_residual, sigma, condition=condition)


@torch.no_grad()
def edm_heun_sample(
    model: StormCastEDMPrecond,
    condition: torch.Tensor,
    output_channels: int,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800.0,
    rho: float = 7.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a residual with the deterministic EDM Heun path."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if not 0 < sigma_min < sigma_max:
        raise ValueError("Expected 0 < sigma_min < sigma_max")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if condition.ndim != 4:
        raise ValueError("condition must have shape (B, C, H, W)")

    step_indices = torch.arange(
        num_steps, device=condition.device, dtype=torch.float64
    )
    denominator = max(num_steps - 1, 1)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / denominator
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ).pow(rho)
    sigma_steps = torch.cat((sigma_steps, sigma_steps.new_zeros(1)))
    shape = (condition.shape[0], output_channels, *condition.shape[-2:])
    latent = torch.randn(
        shape,
        device=condition.device,
        dtype=torch.float32,
        generator=generator,
    )
    x_next = latent.to(torch.float64) * sigma_steps[0]

    for index, (sigma_cur, sigma_next) in enumerate(
        zip(sigma_steps[:-1], sigma_steps[1:])
    ):
        x_cur = x_next
        denoised = model(
            x_cur.to(torch.float32),
            sigma_cur.to(torch.float32),
            condition=condition,
        ).to(torch.float64)
        derivative = (x_cur - denoised) / sigma_cur
        x_next = x_cur + (sigma_next - sigma_cur) * derivative

        if index < num_steps - 1:
            denoised_next = model(
                x_next.to(torch.float32),
                sigma_next.to(torch.float32),
                condition=condition,
            ).to(torch.float64)
            derivative_next = (x_next - denoised_next) / sigma_next
            x_next = x_cur + (sigma_next - sigma_cur) * (
                0.5 * derivative + 0.5 * derivative_next
            )
    return x_next.to(torch.float32)


def build_regression_model(
    image_size: list[int] | tuple[int, int] = (512, 640),
    state_channels: int = 99,
    background_channels: int = 26,
    invariant_channels: int = 2,
    model_channels: int = 128,
    channel_mult: list[int] | tuple[int, ...] = (1, 2, 2, 2, 2),
    attn_resolutions: list[int] | tuple[int, ...] = (),
    **kwargs: Any,
) -> StormCastRegressionUNet:
    return StormCastRegressionUNet(
        img_resolution=list(image_size),
        img_in_channels=state_channels + background_channels + invariant_channels,
        img_out_channels=state_channels,
        model_type="SongUNet",
        model_channels=model_channels,
        channel_mult=list(channel_mult),
        attn_resolutions=list(attn_resolutions),
        embedding_type="zero",
        additive_pos_embed=False,
        **kwargs,
    )


def build_diffusion_model(
    image_size: list[int] | tuple[int, int] = (512, 640),
    state_channels: int = 99,
    invariant_channels: int = 2,
    model_channels: int = 128,
    channel_mult: list[int] | tuple[int, ...] = (1, 2, 2, 2, 2),
    attn_resolutions: list[int] | tuple[int, ...] = (),
    **kwargs: Any,
) -> StormCastEDMPrecond:
    condition_channels = state_channels + state_channels + invariant_channels
    return StormCastEDMPrecond(
        img_resolution=list(image_size),
        img_channels=state_channels + condition_channels,
        img_in_channels=state_channels + condition_channels,
        img_out_channels=state_channels,
        model_type="SongUNet",
        model_channels=model_channels,
        channel_mult=list(channel_mult),
        attn_resolutions=list(attn_resolutions),
        additive_pos_embed=True,
        **kwargs,
    )


def _model_dtype(
    x: torch.Tensor, use_fp16: bool, force_fp32: bool
) -> torch.dtype:
    return (
        torch.float16
        if use_fp16 and not force_fp32 and x.device.type == "cuda"
        else torch.float32
    )


def _validate_image(x: torch.Tensor, name: str, channels: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(x.shape)}")
    if x.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {x.shape[1]}")


def _validate_condition(condition: torch.Tensor, x: torch.Tensor) -> None:
    if condition.ndim != 4:
        raise ValueError("condition must have shape (B, C, H, W)")
    if condition.shape[0] != x.shape[0] or condition.shape[-2:] != x.shape[-2:]:
        raise ValueError("condition batch and spatial dimensions must match x")


def _validate_output_dtype(output: torch.Tensor, dtype: torch.dtype) -> None:
    if output.dtype != dtype and not torch.is_autocast_enabled():
        raise ValueError(f"Expected model output dtype {dtype}, got {output.dtype}")


def _expand_invariant(invariant: torch.Tensor, batch_size: int) -> torch.Tensor:
    if invariant.ndim == 3:
        invariant = invariant.unsqueeze(0)
    if invariant.ndim != 4:
        raise ValueError("invariant must have shape (C, H, W) or (B, C, H, W)")
    if invariant.shape[0] == 1 and batch_size != 1:
        invariant = invariant.expand(batch_size, -1, -1, -1)
    if invariant.shape[0] != batch_size:
        raise ValueError("invariant batch dimension must be 1 or match state")
    return invariant
