import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "result" / "metrics.json"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}. Run scripts/inference.py first.")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print("BENO inference metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
