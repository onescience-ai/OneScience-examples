import math
import warnings

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def get_earth_position_index(window_size, ndim=3):
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        coords_zi = torch.arange(win_pl)
        coords_zj = -torch.arange(win_pl) * win_pl
    elif ndim == 2:
        win_lat, win_lon = window_size
    else:
        raise ValueError("ndim must be 2 or 3")

    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat
    coords_w = torch.arange(win_lon)

    if ndim == 3:
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    else:
        coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))

    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    if ndim == 3:
        coords[:, :, 2] += win_lon - 1
        coords[:, :, 1] *= 2 * win_lon - 1
        coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    else:
        coords[:, :, 1] += win_lon - 1
        coords[:, :, 0] *= 2 * win_lon - 1

    return coords.sum(-1)


def get_pad3d(input_resolution, window_size):
    pressure_levels, height, width = input_resolution
    win_pl, win_height, win_width = window_size

    padding_left = padding_right = padding_top = padding_bottom = 0
    padding_front = padding_back = 0

    pl_remainder = pressure_levels % win_pl
    height_remainder = height % win_height
    width_remainder = width % win_width

    if pl_remainder:
        pressure_pad = win_pl - pl_remainder
        padding_front = pressure_pad // 2
        padding_back = pressure_pad - padding_front
    if height_remainder:
        height_pad = win_height - height_remainder
        padding_top = height_pad // 2
        padding_bottom = height_pad - padding_top
    if width_remainder:
        width_pad = win_width - width_remainder
        padding_left = width_pad // 2
        padding_right = width_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def crop3d(x, resolution):
    _, _, pressure_levels, height, width = x.shape
    pressure_pad = pressure_levels - resolution[0]
    height_pad = height - resolution[1]
    width_pad = width - resolution[2]

    padding_front = pressure_pad // 2
    padding_back = pressure_pad - padding_front
    padding_top = height_pad // 2
    padding_bottom = height_pad - padding_top
    padding_left = width_pad // 2
    padding_right = width_pad - padding_left

    return x[
        :,
        :,
        padding_front : pressure_levels - padding_back,
        padding_top : height - padding_bottom,
        padding_left : width - padding_right,
    ]


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    u1 = norm_cdf((a - mean) / std)
    u2 = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * u1 - 1, 2 * u2 - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def window_partition(x, window_size, ndim=3):
    if ndim == 3:
        batch, pressure_levels, height, width, channels = x.shape
        win_pl, win_height, win_width = window_size
        x = x.view(
            batch,
            pressure_levels // win_pl,
            win_pl,
            height // win_height,
            win_height,
            width // win_width,
            win_width,
            channels,
        )
        return (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(
                -1,
                (pressure_levels // win_pl) * (height // win_height),
                win_pl,
                win_height,
                win_width,
                channels,
            )
        )

    if ndim == 2:
        batch, height, width, channels = x.shape
        win_height, win_width = window_size
        x = x.view(batch, height // win_height, win_height, width // win_width, win_width, channels)
        return (
            x.permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(-1, height // win_height, win_height, win_width, channels)
        )

    raise ValueError("ndim must be 2 or 3")


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    if ndim == 3:
        win_pl, win_height, win_width = window_size
        batch = int(windows.shape[0] / (Lon / win_width))
        x = windows.view(
            batch,
            Lon // win_width,
            Pl // win_pl,
            Lat // win_height,
            win_pl,
            win_height,
            win_width,
            -1,
        )
        return (
            x.permute(0, 2, 4, 3, 5, 1, 6, 7)
            .contiguous()
            .view(batch, Pl, Lat, Lon, -1)
        )

    if ndim == 2:
        win_height, win_width = window_size
        batch = int(windows.shape[0] / (Lon / win_width))
        x = windows.view(batch, Lon // win_width, Lat // win_height, win_height, win_width, -1)
        return x.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch, Lat, Lon, -1)

    raise ValueError("ndim must be 2 or 3")


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    if ndim == 3:
        pressure_levels, height, width = input_resolution
        win_pl, win_height, win_width = window_size
        shift_pl, shift_height, shift_width = shift_size
        img_mask = torch.zeros((1, pressure_levels, height, width + shift_width, 1))
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    elif ndim == 2:
        height, width = input_resolution
        win_height, win_width = window_size
        shift_height, shift_width = shift_size
        img_mask = torch.zeros((1, height, width + shift_width, 1))
    else:
        raise ValueError("ndim must be 2 or 3")

    height_slices = (
        slice(0, -win_height),
        slice(-win_height, -shift_height),
        slice(-shift_height, None),
    )
    width_slices = (
        slice(0, -win_width),
        slice(-win_width, -shift_width),
        slice(-shift_width, None),
    )

    count = 0
    if ndim == 3:
        for pressure_slice in pl_slices:
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, pressure_slice, height_slice, width_slice, :] = count
                    count += 1
        img_mask = img_mask[:, :, :, :width, :]
    else:
        for height_slice in height_slices:
            for width_slice in width_slices:
                img_mask[:, height_slice, width_slice, :] = count
                count += 1
        img_mask = img_mask[:, :, :width, :]

    mask_windows = window_partition(img_mask, window_size, ndim=ndim)
    win_total = win_pl * win_height * win_width if ndim == 3 else win_height * win_width
    mask_windows = mask_windows.view(mask_windows.shape[0], mask_windows.shape[1], win_total)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
