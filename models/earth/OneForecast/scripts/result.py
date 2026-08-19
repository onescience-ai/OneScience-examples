"""Create quick field images from OneForecast prediction files."""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = args.config.resolve().parent.parent
    input_dir = Path(config["visualization"]["input_dir"])
    output_dir = Path(config["visualization"]["output_dir"])
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("prediction_*.npy"))
    if not files:
        raise SystemExit(f"No prediction files found in {input_dir}")
    import matplotlib.pyplot as plt

    channels = config["visualization"].get("channels", [0])
    for source in files:
        prediction = np.load(source)
        if prediction.shape != (1, 69, 120, 240):
            raise ValueError(f"Expected official prediction shape [1, 69, 120, 240], got {prediction.shape}")
        field = prediction[0]
        for channel in channels:
            if channel < 0 or channel >= field.shape[0]:
                raise ValueError(f"Channel {channel} is outside prediction shape {field.shape}")
            figure, axis = plt.subplots(figsize=(8, 3.5))
            image = axis.imshow(field[channel], cmap="coolwarm", aspect="auto")
            axis.set_title(f"{source.stem}, channel {channel}")
            axis.set_xlabel("longitude index")
            axis.set_ylabel("latitude index")
            figure.colorbar(image, ax=axis, shrink=0.8)
            figure.tight_layout()
            figure.savefig(output_dir / f"{source.stem}_ch{channel}.png", dpi=160)
            plt.close(figure)
    print({"input_dir": str(input_dir), "output_dir": str(output_dir), "files": len(files)})


if __name__ == "__main__":
    main()
