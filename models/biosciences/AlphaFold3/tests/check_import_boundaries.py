from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = (
    PROJECT_ROOT / "flax_model",
    PROJECT_ROOT / "model",
    PROJECT_ROOT / "scripts",
)


def _python_files(root: Path):
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _check_no_onescience_model_imports():
    pattern = re.compile(
        r"onescience\.(flax_models|models)\.",
        re.MULTILINE,
    )
    offenders = []
    for root in LOCAL_ROOTS:
        for path in _python_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    return offenders


def main():
    offenders = _check_no_onescience_model_imports()
    if offenders:
        print("Forbidden onescience model imports:")
        for offender in offenders:
            print(f"  {offender}")
        return 1

    print("Import boundary checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
