from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


def _python_files(root: Path):
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _check_no_onescience_models_imports():
    offenders = []
    for path in _python_files(MODEL_ROOT):
        text = path.read_text(encoding="utf-8")
        if "onescience.models." in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    return offenders


def _check_no_legacy_imports():
    pattern = re.compile(
        r"^\s*(from\s+(esm|openfold)(\.|\s)|import\s+(esm|openfold)(\.|\s|$))",
        re.MULTILINE,
    )
    offenders = []
    for root in (MODEL_ROOT, SCRIPT_ROOT):
        for path in _python_files(root):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    return offenders


def _check_script_bootstrap():
    offenders = []
    for path in _python_files(SCRIPT_ROOT):
        text = path.read_text(encoding="utf-8")
        if "model.esm" in text and '_PROJECT_ROOT / "model"' not in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    return offenders


def main():
    checks = {
        "onescience.models imports": _check_no_onescience_models_imports(),
        "legacy esm/openfold imports": _check_no_legacy_imports(),
        "script project-root bootstrap": _check_script_bootstrap(),
    }
    failed = {name: offenders for name, offenders in checks.items() if offenders}
    if failed:
        for name, offenders in failed.items():
            print(f"{name}:")
            for offender in offenders:
                print(f"  {offender}")
        return 1

    print("Import boundary checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

