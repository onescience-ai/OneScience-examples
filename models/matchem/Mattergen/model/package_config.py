from pathlib import Path


def get_package_data():
    package = "model"
    data = {
        package: ["LICENSE", "NOTICE", "SOURCE.md", "**/*.yaml", "**/*.json"],
    }
    return data
