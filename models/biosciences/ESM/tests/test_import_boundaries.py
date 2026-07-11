from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


def _python_files(root: Path):
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def test_local_model_code_does_not_import_onescience_models_namespace():
    offenders = []
    for path in _python_files(MODEL_ROOT):
        text = path.read_text(encoding="utf-8")
        if "onescience.models." in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_python_files_do_not_use_legacy_top_level_esm_or_openfold_imports():
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

    assert offenders == []


def test_scripts_bootstrap_project_root_before_importing_local_model():
    offenders = []
    for path in _python_files(SCRIPT_ROOT):
        text = path.read_text(encoding="utf-8")
        if "model.esm" in text and '_PROJECT_ROOT / "model"' not in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []

