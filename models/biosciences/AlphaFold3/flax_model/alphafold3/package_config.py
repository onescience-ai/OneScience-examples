#!/usr/bin/env python3
"""Package metadata helpers for the split AlphaFold3 module."""

ALPHAFOLD3_PACKAGE_DATA = {
    "flax_model.alphafold3": [
        "*.so",
        "*.pyd",
        "*.dll",
        "*.dylib",
        "README.md",
        "test_data/**/*",
        "**/*.pyi",
    ],
    "flax_model.alphafold3.constants.converters": [
        "*.pickle",
    ],
}

ALPHAFOLD3_MANIFEST_RULES = [
    "include flax_model/alphafold3/*.so",
    "include flax_model/alphafold3/*.pyd",
    "include flax_model/alphafold3/*.dll",
    "include flax_model/alphafold3/*.dylib",
    "recursive-include flax_model/alphafold3/constants/converters *.pickle",
    "include flax_model/alphafold3/README.md",
    "recursive-include flax_model/alphafold3/test_data *",
]


def get_package_data():
    return ALPHAFOLD3_PACKAGE_DATA


def get_manifest_rules():
    return ALPHAFOLD3_MANIFEST_RULES


_build_done = False


def build_hook(project_root, env, python_executable, subprocess_module):
    del project_root, env, python_executable, subprocess_module
    global _build_done
    if _build_done:
        return

    from flax_model.alphafold3 import _build as af3_build

    print("[AF3] local build hook triggered")
    try:
        print(
            "[AF3] "
            f"should_build={af3_build.should_build()} "
            f"artifacts_exist={af3_build.artifacts_exist()} "
            f"force_rebuild={af3_build.force_rebuild()}"
        )
        af3_build.build_if_needed()
        print("[AF3] local build hook finished")
    except af3_build.AF3BuildError as exc:
        print(f"[AF3] local build skipped or failed: {exc}")
        if af3_build.is_strict():
            raise

    _build_done = True


def get_build_hook():
    return build_hook
