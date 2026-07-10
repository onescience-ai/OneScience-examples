from typing import Tuple

import torch

from model.auto_deeponet import AutoDeepONet
from model.auto_deeponet_cnn import AutoDeepONetCnn
from model.auto_edeeponet import AutoEDeepONet
from model.auto_ffn import AutoFfn
from model.deeponet import DeepONet
from model.ffn import FfnModel
from model.fno.fno2d import Fno2d
from model.loss import loss_name_to_fn
from model.resnet import ResNet
from model.unet import UNet


AUTO_MODEL_NAMES = {
    "auto_ffn",
    "auto_deeponet",
    "auto_edeeponet",
    "auto_deeponet_cnn",
    "resnet",
    "unet",
    "fno",
}
STATIC_MODEL_NAMES = {"ffn", "deeponet"}
ALL_MODEL_NAMES = AUTO_MODEL_NAMES | STATIC_MODEL_NAMES


def infer_task_type(model_name: str) -> str:
    if model_name in AUTO_MODEL_NAMES:
        return "auto"
    if model_name in STATIC_MODEL_NAMES:
        return "static"
    raise ValueError(f"Unknown CFDBench model.name={model_name!r}. Available: {sorted(ALL_MODEL_NAMES)}")


def get_input_shapes(data_name: str, num_rows: int, num_cols: int) -> Tuple[int, int, int]:
    if any(name in data_name for name in ("tube", "dam", "cylinder")):
        n_rows = num_rows + 2
        n_cols = num_cols + 1
    elif "cavity" in data_name:
        n_rows = num_rows
        n_cols = num_cols
    else:
        raise ValueError(f"Unknown CFDBench problem in data_name: {data_name}")

    if "cylinder" in data_name:
        n_case_params = 8
    elif any(name in data_name for name in ("cavity", "tube", "dam")):
        n_case_params = 5
    else:
        raise ValueError(f"Unknown CFDBench parameter set in data_name: {data_name}")

    return n_rows, n_cols, n_case_params


def build_model(cfg) -> torch.nn.Module:
    model_cfg = cfg.model
    data_cfg = cfg.datapipe.data
    source_cfg = cfg.datapipe.source
    train_cfg = cfg.training
    loss_fn = loss_name_to_fn(train_cfg.loss_name)
    n_rows, n_cols, n_case_params = get_input_shapes(
        data_name=source_cfg.data_name,
        num_rows=data_cfg.num_rows,
        num_cols=data_cfg.num_cols,
    )

    name = model_cfg.name
    if name == "ffn":
        widths = [n_case_params + 3] + [model_cfg.ffn_width] * model_cfg.ffn_depth + [1]
        return FfnModel(
            widths=widths,
            loss_fn=loss_fn,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            act_on_output=model_cfg.act_on_output,
            num_label_samples=model_cfg.num_label_samples,
        )
    if name == "deeponet":
        return DeepONet(
            branch_dim=n_case_params,
            trunk_dim=3,
            loss_fn=loss_fn,
            width=model_cfg.deeponet_width,
            trunk_depth=model_cfg.trunk_depth,
            branch_depth=model_cfg.branch_depth,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            act_on_output=model_cfg.act_on_output,
            num_label_samples=model_cfg.num_label_samples,
        )
    if name == "auto_ffn":
        return AutoFfn(
            input_field_dim=n_rows * n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
            width=model_cfg.autoffn_width,
            depth=model_cfg.autoffn_depth,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            num_label_samples=model_cfg.num_label_samples,
        )
    if name == "auto_deeponet":
        return AutoDeepONet(
            branch_dim=n_rows * n_cols + n_case_params,
            trunk_dim=2,
            loss_fn=loss_fn,
            width=model_cfg.deeponet_width,
            trunk_depth=model_cfg.trunk_depth,
            branch_depth=model_cfg.branch_depth,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            act_on_output=model_cfg.act_on_output,
            num_label_samples=model_cfg.num_label_samples,
        )
    if name == "auto_edeeponet":
        return AutoEDeepONet(
            dim_branch1=n_rows * n_cols,
            dim_branch2=n_case_params,
            trunk_dim=2,
            loss_fn=loss_fn,
            width=model_cfg.autoedeeponet_width,
            trunk_depth=model_cfg.autoedeeponet_depth,
            branch_depth=model_cfg.autoedeeponet_depth,
            act_name=model_cfg.autoedeeponet_act_fn,
            num_label_samples=model_cfg.num_label_samples,
        )
    if name == "auto_deeponet_cnn":
        if n_rows // 16 != 4 or n_cols // 16 != 4:
            raise ValueError(
                "auto_deeponet_cnn expects the post-padding grid to reduce to 4x4 after four 2x pools. "
                "For tube/dam/cylinder fake data, set datapipe.data.num_rows=64 and num_cols=64."
            )
        return AutoDeepONetCnn(
            in_chan=model_cfg.in_chan,
            height=n_rows,
            width=n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
            trunk_depth=model_cfg.trunk_depth,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            act_on_output=model_cfg.act_on_output,
        )
    if name == "resnet":
        return ResNet(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            n_case_params=n_case_params,
            loss_fn=loss_fn,
            hidden_chan=model_cfg.resnet_hidden_chan,
            num_blocks=model_cfg.resnet_depth,
            kernel_size=model_cfg.resnet_kernel_size,
            padding=model_cfg.resnet_padding,
        )
    if name == "unet":
        return UNet(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            insert_case_params_at=model_cfg.unet_insert_case_params_at,
            dim=model_cfg.unet_dim,
        )
    if name == "fno":
        return Fno2d(
            in_chan=model_cfg.in_chan,
            out_chan=model_cfg.out_chan,
            n_case_params=n_case_params,
            loss_fn=loss_fn,
            num_layers=model_cfg.fno_depth,
            hidden_dim=model_cfg.fno_hidden_dim,
            modes1=model_cfg.fno_modes_x,
            modes2=model_cfg.fno_modes_y,
        )

    raise ValueError(f"Invalid CFDBench model.name={name!r}. Available: {sorted(ALL_MODEL_NAMES)}")
