#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：备份模型目录、重命名、清理文件、同步到魔搭、生成 download.sh、删除本地大文件。

默认流程：
1. 备份 --model-dir 指向的原始模型文件夹：
       原文件夹名_backup
2. 将模型文件夹重命名为 --model 设置的名称。
3. 删除 .doc / .docx 文件，删除前需要确认。
4. 上传/同步前删除 .ipynb_checkpoints 文件夹，并清理 .ms_upload_cache。
5. 同步到魔搭社区，支持两种模式：
       upload: 使用 modelscope upload，适合同名覆盖或追加文件。
       git: 使用 Git 精确同步，适合删除、重命名、目录结构调整。
6. 同步成功后再次清理 .ms_upload_cache，然后生成 download.sh。
7. 删除 >1MB 且扩展名不是 .sh/.py/.md/.yaml/.yml 的本地大文件，删除前需要确认。

参数释义：
--model  # 模型名，例如 hweh；模型文件夹最终会改名为该名称
--model-dir  # 本地模型文件夹路径
--backup-dir  # 备份文件夹路径；默认是模型文件夹同级目录下的 原文件夹名_backup
--namespace  # 命名空间，默认 OneScience
--token  # 魔搭 token；也可以用环境变量“export MODELSCOPE_TOKEN=你的token”指定
--sync-mode  # 同步方式，默认 upload；可选 upload 或 git
--repo-url  # Git 同步模式下的魔搭仓库地址；默认自动生成 https://www.modelscope.cn/<namespace>/<model>.git
--commit-message  # Git 同步模式下的提交信息，默认 sync model files
--keep-git-workdir  # Git 同步失败排查用：保留临时 clone 工作区
--overwrite-backup  # 如果备份目录已存在，先删除旧备份再重新备份
--remove-empty-dirs  # 删除大文件后同步删除空目录
--yes  # 跳过删除确认，直接删除 .doc/.docx、.ipynb_checkpoints 和大文件
--dry-run  # 只预览流程，不创建备份、不重命名、不删除文件、不上传、不生成 download.sh

示例：
    export MODELSCOPE_TOKEN=你的token
    python modelscope_upload_cleanup.py --model hweh --model-dir /path_to_file/hweh_446k_test

Git 精确同步示例：
    export MODELSCOPE_TOKEN=你的token
    python modelscope_upload_cleanup.py --model hweh --model-dir /path_to_file/hweh_446k_test --sync-mode git

跳过删除确认：
    python modelscope_upload_cleanup.py --model hweh --model-dir /path_to_file/hweh_446k_test --yes
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_NAMESPACE = "OneScience"
TOKEN_ENV_VAR = "MODELSCOPE_TOKEN"
CHECKPOINT_DIR_NAME = ".ipynb_checkpoints"
UPLOAD_CACHE_NAME = ".ms_upload_cache"
DOC_EXTENSIONS = {".doc", ".docx"}
LARGE_FILE_EXCLUDE_EXT = {".sh", ".py", ".md", ".yaml", ".yml"}
SIZE_THRESHOLD = 1024 * 1024
LFS_PATTERNS = [
    "*.bin",
    "*.ckpt",
    "*.h5",
    "*.hdf5",
    "*.joblib",
    "*.model",
    "*.msgpack",
    "*.npy",
    "*.npz",
    "*.onnx",
    "*.ot",
    "*.parquet",
    "*.pkl",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.tflite",
    "*.zip",
]


def ask(prompt: str, default: str | None = None, required: bool = True) -> str:
    if default:
        text = input(f"{prompt} [{default}]: ").strip()
        return text or default

    while True:
        text = input(f"{prompt}: ").strip()
        if text or not required:
            return text
        print("该项不能为空，请重新输入。")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "yes" if default else "no"
    while True:
        text = input(f"{prompt} (yes/no) [{default_text}]: ").strip().lower()
        if not text:
            return default
        if text in {"y", "yes"}:
            return True
        if text in {"n", "no"}:
            return False
        print("请输入 yes 或 no。")


def mask_token(token: str) -> str:
    if len(token) <= 14:
        return "<TOKEN>"
    return f"{token[:8]}...{token[-6:]}"


def resolve_token(cli_token: str | None) -> tuple[str, str]:
    if cli_token:
        return cli_token.strip(), "--token"

    env_token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if env_token:
        return env_token, f"环境变量 {TOKEN_ENV_VAR}"

    raise RuntimeError(f"未找到 token。请使用 --token 指定，或先执行：export {TOKEN_ENV_VAR}=你的魔搭token")


def validate_model_name(model_name: str) -> None:
    if not model_name:
        raise ValueError("模型名不能为空。")
    if "/" in model_name or "\\" in model_name:
        raise ValueError("模型名不能包含 / 或 \\。请只填写 OneScience/ 后面的模型名。")


def resolve_existing_dir(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label}不是文件夹: {path}")
    return path


def default_backup_path(model_dir: Path) -> Path:
    return model_dir.parent / f"{model_dir.name}_backup"


def default_repo_url(namespace: str, model_name: str) -> str:
    return f"https://www.modelscope.cn/{namespace}/{model_name}.git"


def print_backup_location(path: Path, dry_run: bool, prefix: str = "") -> None:
    label = "预计备份位置" if dry_run else "备份位置"
    print(f"{prefix}{label}: {path}")


def command_text(command: list[str], hidden_values: set[str] | None = None) -> str:
    hidden_values = hidden_values or set()
    rendered = []
    for part in command:
        text = str(part)
        for hidden in hidden_values:
            if hidden:
                text = text.replace(hidden, "<TOKEN>")
        rendered.append(shlex.quote(text))
    return " ".join(rendered)


def run_command(
    command: list[str],
    cwd: Path | None = None,
    token: str | None = None,
    dry_run: bool = False,
    label: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    hidden = {token} if token else set()
    if label:
        print(f"\n{label}")
    print(f"  {command_text(command, hidden)}")

    if dry_run:
        print("当前为 dry-run，不执行该命令。")
        return None

    try:
        return subprocess.run(command, cwd=cwd, env=env, text=True, check=False)
    except FileNotFoundError:
        print(f"错误：找不到命令 {command[0]}。请先安装并确认它在 PATH 中。", file=sys.stderr)
        return subprocess.CompletedProcess(command, 127)


def backup_model_dir(src: Path, backup_path: Path, overwrite_backup: bool, dry_run: bool = False) -> Path:
    if dry_run:
        raise RuntimeError("内部错误：dry-run 模式不允许创建备份。")

    backup_path = backup_path.expanduser().resolve()
    if backup_path == src:
        raise ValueError("备份目录不能和模型目录相同。")
    if src in backup_path.parents:
        raise ValueError("备份目录不能放在模型目录内部。")

    if backup_path.exists():
        if not overwrite_backup:
            raise FileExistsError(
                f"备份目录已存在: {backup_path}\n"
                "请先删除或改名已有备份目录，或使用 --overwrite-backup 允许重建备份。"
            )
        print(f"\n[1/7] 备份目录已存在，将先删除旧备份: {backup_path}")
        shutil.rmtree(backup_path)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/7] 开始备份:\n  来源: {src}\n  目标: {backup_path}")
    shutil.copytree(src, backup_path)
    print("备份完成。")
    return backup_path


def rename_model_dir(model_dir: Path, model_name: str) -> Path:
    target_dir = model_dir.parent / model_name
    if model_dir == target_dir:
        print(f"\n[2/7] 模型目录名称已是 --model 指定名称: {model_dir.name}")
        return model_dir

    if target_dir.exists():
        raise FileExistsError(
            f"重命名目标目录已存在: {target_dir}\n"
            "请先处理该目录，或换一个 --model 名称。脚本不会覆盖已有模型目录。"
        )

    print(f"\n[2/7] 重命名模型目录:\n  原目录: {model_dir}\n  新目录: {target_dir}")
    model_dir.rename(target_dir)
    print("重命名完成。")
    return target_dir


def find_doc_files(model_dir: Path) -> list[Path]:
    return sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS)


def delete_doc_files(model_dir: Path, assume_yes: bool, dry_run: bool) -> bool:
    doc_files = find_doc_files(model_dir)
    if not doc_files:
        print("\n[3/7] 没有找到 .doc / .docx 文件。")
        return True

    print(f"\n[3/7] 将删除 {len(doc_files)} 个 .doc / .docx 文件:")
    for file_path in doc_files:
        print(f"  {file_path.relative_to(model_dir)}")

    if dry_run:
        print("当前为 dry-run，不删除 .doc / .docx 文件。")
        return True

    if not assume_yes and not ask_yes_no("确认删除这些 .doc / .docx 文件吗？", default=False):
        print("已取消删除 .doc / .docx 文件，流程停止，未上传。")
        return False

    for file_path in doc_files:
        file_path.unlink()
        print(f"已删除: {file_path.relative_to(model_dir)}")
    print(f"共删除 {len(doc_files)} 个文档文件。")
    return True


def find_named_paths(model_dir: Path, name: str) -> list[Path]:
    return sorted((p for p in model_dir.rglob(name) if p.name == name), key=lambda p: len(p.parts), reverse=True)


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def cleanup_named_paths(
    model_dir: Path,
    name: str,
    label: str,
    assume_yes: bool,
    dry_run: bool,
    confirm: bool,
    step: str,
    stop_message: str | None = None,
) -> bool:
    paths = find_named_paths(model_dir, name)
    if not paths:
        print(f"\n[{step}] 未找到 {label}。")
        return True

    print(f"\n[{step}] 将删除 {len(paths)} 个 {label}:")
    for path in paths:
        print(f"  {path.relative_to(model_dir)}")

    if dry_run:
        print(f"当前为 dry-run，不删除 {label}。")
        return True

    if confirm and not assume_yes and not ask_yes_no(f"确认删除这些 {label} 吗？", default=False):
        print(stop_message or f"已取消删除 {label}。")
        return False

    deleted = 0
    for path in paths:
        try:
            rel_path = path.relative_to(model_dir)
            delete_path(path)
            deleted += 1
            print(f"已删除: {rel_path}")
        except OSError as exc:
            print(f"删除失败 {path}: {exc}")

    print(f"共删除 {deleted} 个 {label}。")
    return True


def run_modelscope_upload(namespace: str, model_name: str, model_dir: Path, token: str, dry_run: bool) -> int:
    command = ["modelscope", "upload", f"{namespace}/{model_name}", str(model_dir), "--token", token]
    result = run_command(command, token=token, dry_run=dry_run, label="\n[5/7] modelscope upload 上传命令:")
    if dry_run:
        return 0
    if result is None:
        return 1
    if result.returncode == 0:
        print("上传完成。")
    else:
        print(f"上传失败，状态码: {result.returncode}", file=sys.stderr)
    return result.returncode


def git_auth_repo_url(repo_url: str, token: str) -> str:
    if repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://oauth2:{token}@", 1)
    return repo_url


def copy_public_files_to_repo(src_dir: Path, repo_dir: Path) -> None:
    for child in repo_dir.iterdir():
        if child.name == ".git":
            continue
        delete_path(child)

    for child in src_dir.iterdir():
        if child.name == ".git":
            continue
        target = repo_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def configure_git_identity(repo_dir: Path, token: str, dry_run: bool) -> int:
    commands = [
        ["git", "config", "user.name", "modelscope-upload-script"],
        ["git", "config", "user.email", "modelscope-upload-script@example.com"],
    ]
    for command in commands:
        result = run_command(command, cwd=repo_dir, token=token, dry_run=dry_run)
        if result is not None and result.returncode != 0:
            return result.returncode
    return 0


def configure_git_lfs(repo_dir: Path, token: str, dry_run: bool) -> int:
    result = run_command(["git", "lfs", "install", "--local"], cwd=repo_dir, token=token, dry_run=dry_run)
    if result is not None and result.returncode != 0:
        return result.returncode

    for pattern in LFS_PATTERNS:
        result = run_command(["git", "lfs", "track", pattern], cwd=repo_dir, token=token, dry_run=dry_run)
        if result is not None and result.returncode != 0:
            return result.returncode
    return 0


def git_has_changes(repo_dir: Path, token: str) -> bool:
    result = run_command(["git", "status", "--porcelain"], cwd=repo_dir, token=token)
    if result is None:
        return False
    return result.stdout.strip() != "" if result.stdout is not None else True


def run_git_sync(
    repo_url: str,
    model_dir: Path,
    token: str,
    commit_message: str,
    dry_run: bool,
    keep_git_workdir: bool,
) -> int:
    auth_url = git_auth_repo_url(repo_url, token)
    print(f"\n[5/7] Git 精确同步仓库:\n  仓库: {repo_url}")

    if dry_run:
        print("当前为 dry-run，不执行 git clone / commit / push。")
        print("将使用 git add -A 记录新增、修改、删除和重命名。")
        return 0

    temp_obj = tempfile.TemporaryDirectory(prefix="modelscope_git_sync_")
    temp_root = Path(temp_obj.name)
    repo_dir = temp_root / "repo"

    try:
        result = run_command(["git", "clone", auth_url, str(repo_dir)], token=token, label="克隆魔搭仓库:")
        if result is None or result.returncode != 0:
            return 1 if result is None else result.returncode

        code = configure_git_identity(repo_dir, token, dry_run=False)
        if code != 0:
            return code

        code = configure_git_lfs(repo_dir, token, dry_run=False)
        if code != 0:
            return code

        copy_public_files_to_repo(model_dir, repo_dir)

        result = run_command(["git", "add", "-A"], cwd=repo_dir, token=token, label="暂存变更:")
        if result is None or result.returncode != 0:
            return 1 if result is None else result.returncode

        result = run_command(["git", "status", "--short"], cwd=repo_dir, token=token, label="Git 变更预览:")
        if result is None or result.returncode != 0:
            return 1 if result is None else result.returncode

        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir, text=True, check=False)
        if result.returncode == 0:
            print("没有检测到需要同步的 Git 变更。")
            return 0
        if result.returncode not in {0, 1}:
            return result.returncode

        result = run_command(["git", "commit", "-m", commit_message], cwd=repo_dir, token=token, label="提交变更:")
        if result is None or result.returncode != 0:
            return 1 if result is None else result.returncode

        result = run_command(["git", "push"], cwd=repo_dir, token=token, label="推送到魔搭:")
        if result is None or result.returncode != 0:
            return 1 if result is None else result.returncode

        print("Git 精确同步完成。")
        return 0
    finally:
        if keep_git_workdir:
            print(f"已保留临时 Git 工作区: {temp_root}")
        else:
            temp_obj.cleanup()


def is_under_skipped_dir(path: Path) -> bool:
    return CHECKPOINT_DIR_NAME in path.parts or UPLOAD_CACHE_NAME in path.parts or ".git" in path.parts


def is_large_download_target(path: Path) -> bool:
    return (
        path.is_file()
        and not is_under_skipped_dir(path)
        and path.stat().st_size > SIZE_THRESHOLD
        and path.suffix.lower() not in LARGE_FILE_EXCLUDE_EXT
    )


def find_large_files(model_dir: Path) -> list[Path]:
    return sorted(p for p in model_dir.rglob("*") if is_large_download_target(p))


def generate_download_script(
    model_dir: Path,
    namespace: str,
    model_name: str,
    large_files: list[Path],
    dry_run: bool,
    relative_root: Path | None = None,
) -> Path | None:
    if not large_files:
        print("\n[6/7] 没有需要写入 download.sh 的大文件。")
        return None

    commands = []
    relative_root = relative_root or model_dir
    for file_path in large_files:
        rel_path = file_path.relative_to(relative_root).as_posix()
        commands.append(
            "modelscope download "
            f"--model {shlex.quote(namespace + '/' + model_name)} "
            f"{shlex.quote(rel_path)} "
            "--local_dir ./"
        )

    if dry_run:
        print(f"\n[6/7] dry-run：将生成 download.sh，共 {len(commands)} 条下载命令:")
        for command in commands:
            print(f"  {command}")
        return None

    script_path = model_dir / "download.sh"
    with script_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write("# Download files larger than 1MB, excluding .sh .py .md .yaml .yml\n")
        f.write(f"# Total large files: {len(large_files)}\n")
        for command in commands:
            f.write(command + "\n")

    try:
        os.chmod(script_path, 0o755)
    except OSError:
        pass

    print(f"\n[6/7] 已生成 {script_path}，共 {len(large_files)} 条下载命令。")
    return script_path


def print_large_file_preview(model_dir: Path, large_files: list[Path], relative_root: Path | None = None) -> None:
    total_bytes = sum(file_path.stat().st_size for file_path in large_files)
    relative_root = relative_root or model_dir
    print(f"\n[7/7] 将删除 {len(large_files)} 个大文件，合计约 {total_bytes / (1024 ** 3):.2f} GB:")
    for file_path in large_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {file_path.relative_to(relative_root)}  ({size_mb:.2f} MB)")


def remove_empty_dirs(root_dir: Path) -> int:
    removed = 0
    dirs = sorted((p for p in root_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for directory in dirs:
        try:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                removed += 1
                print(f"已删除空目录: {directory}")
        except OSError as exc:
            print(f"删除空目录失败 {directory}: {exc}")
    return removed


def delete_large_files(
    model_dir: Path,
    large_files: list[Path],
    assume_yes: bool,
    remove_empty: bool,
    dry_run: bool,
    relative_root: Path | None = None,
) -> int:
    if not large_files:
        print("\n[7/7] 没有需要删除的大文件。")
        return 0

    relative_root = relative_root or model_dir
    print_large_file_preview(model_dir, large_files, relative_root=relative_root)

    if dry_run:
        print("当前为 dry-run，不删除大文件。")
        return 0

    if not assume_yes and not ask_yes_no("确认删除以上大文件吗？", default=False):
        print("已取消删除大文件。")
        return 0

    deleted = 0
    for file_path in large_files:
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_path.unlink()
            deleted += 1
            print(f"已删除: {file_path.relative_to(relative_root)} ({size_mb:.2f} MB)")
        except OSError as exc:
            print(f"删除失败 {file_path}: {exc}")

    if remove_empty:
        remove_empty_dirs(model_dir)

    print(f"共删除 {deleted} 个大文件。")
    return deleted


def preview_dry_run(
    args: argparse.Namespace,
    model_name: str,
    original_model_dir: Path,
    target_model_dir: Path,
    backup_path: Path,
    repo_url: str,
    token: str,
) -> int:
    print(f"\n[1/7] dry-run：将备份目录:\n  来源: {original_model_dir}\n  目标: {backup_path}")
    if backup_path.exists() and not args.overwrite_backup:
        print("  提示：备份目录已存在；实际运行时需要处理该目录或添加 --overwrite-backup。")

    print(f"\n[2/7] dry-run：将重命名模型目录:\n  原目录: {original_model_dir}\n  新目录: {target_model_dir}")
    if target_model_dir.exists() and original_model_dir != target_model_dir:
        print("  提示：目标目录已存在；实际运行时会停止，脚本不会覆盖已有模型目录。")

    delete_doc_files(original_model_dir, assume_yes=args.yes, dry_run=True)
    cleanup_named_paths(
        model_dir=original_model_dir,
        name=CHECKPOINT_DIR_NAME,
        label=".ipynb_checkpoints 文件夹",
        assume_yes=args.yes,
        dry_run=True,
        confirm=True,
        step="4/7",
        stop_message="已取消删除 .ipynb_checkpoints，流程停止，未同步。",
    )
    cleanup_named_paths(
        model_dir=original_model_dir,
        name=UPLOAD_CACHE_NAME,
        label=".ms_upload_cache",
        assume_yes=True,
        dry_run=True,
        confirm=False,
        step="4/7",
    )

    if args.sync_mode == "git":
        run_git_sync(
            repo_url=repo_url,
            model_dir=original_model_dir,
            token=token,
            commit_message=args.commit_message,
            dry_run=True,
            keep_git_workdir=args.keep_git_workdir,
        )
    else:
        run_modelscope_upload(
            namespace=args.namespace,
            model_name=model_name,
            model_dir=target_model_dir,
            token=token,
            dry_run=True,
        )

    cleanup_named_paths(
        model_dir=original_model_dir,
        name=UPLOAD_CACHE_NAME,
        label=".ms_upload_cache",
        assume_yes=True,
        dry_run=True,
        confirm=False,
        step="6/7",
    )

    large_files = find_large_files(original_model_dir)
    generate_download_script(
        model_dir=original_model_dir,
        namespace=args.namespace,
        model_name=model_name,
        large_files=large_files,
        dry_run=True,
        relative_root=original_model_dir,
    )
    delete_large_files(
        model_dir=original_model_dir,
        large_files=large_files,
        assume_yes=args.yes,
        remove_empty=args.remove_empty_dirs,
        dry_run=True,
        relative_root=original_model_dir,
    )

    print("\ndry-run 预览完成，没有执行任何文件系统写入、上传或 Git 操作。")
    print_backup_location(backup_path, dry_run=True, prefix="  ")
    print(f"  预计模型目录: {target_model_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="备份、重命名、清理、同步魔搭、生成 download.sh 并删除大文件。")
    parser.add_argument("--model", help="模型名，例如 hweh；模型文件夹最终会改名为该名称")
    parser.add_argument("--model-dir", help="本地模型文件夹路径，例如 /root/private_data/guancl/upmodels/hweh_446k_test")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help=f"命名空间，默认 {DEFAULT_NAMESPACE}")
    parser.add_argument("--token", help=f"魔搭 token；也可以用环境变量 {TOKEN_ENV_VAR} 指定")
    parser.add_argument("--sync-mode", choices=["upload", "git"], default="upload", help="同步方式：upload 使用 modelscope upload；git 使用 Git 精确同步")
    parser.add_argument("--repo-url", help="Git 同步模式下的魔搭仓库地址；默认自动生成 https://www.modelscope.cn/<namespace>/<model>.git")
    parser.add_argument("--commit-message", default="sync model files", help="Git 同步模式下的提交信息")
    parser.add_argument("--keep-git-workdir", action="store_true", help="Git 同步失败排查用：保留临时 clone 工作区")
    parser.add_argument("--backup-dir", help="备份文件夹路径；默认是模型文件夹同级目录下的 原文件夹名_backup")
    parser.add_argument("--overwrite-backup", action="store_true", help="如果备份目录已存在，先删除旧备份再重新备份")
    parser.add_argument("--remove-empty-dirs", action="store_true", help="删除大文件后同步删除空目录")
    parser.add_argument("--yes", action="store_true", help="跳过删除确认，直接删除 .doc/.docx、.ipynb_checkpoints 和大文件")
    parser.add_argument("--dry-run", action="store_true", help="只预览流程，不创建备份、不重命名、不删除文件、不上传、不生成 download.sh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_name = args.model or ask("请输入模型名，例如 hweh")
    try:
        validate_model_name(model_name)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    model_dir_raw = args.model_dir or ask("请输入本地模型文件夹路径")
    try:
        original_model_dir = resolve_existing_dir(model_dir_raw, "本地模型文件夹")
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    backup_path = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else default_backup_path(original_model_dir)
    repo_url = args.repo_url or default_repo_url(args.namespace, model_name)

    try:
        token, token_source = resolve_token(args.token)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    target_model_dir = original_model_dir.parent / model_name

    print("\n请确认配置:")
    print(f"  模型仓库: {args.namespace}/{model_name}")
    print(f"  原始目录: {original_model_dir}")
    print(f"  目标目录: {target_model_dir}")
    print(f"  备份目录: {backup_path}")
    print(f"  同步模式: {args.sync_mode}")
    if args.sync_mode == "git":
        print(f"  Git 仓库: {repo_url}")
    print(f"  token来源: {token_source}")
    print(f"  token: {mask_token(token)}")
    print(f"  删除确认: {'跳过（--yes）' if args.yes else '需要手动确认'}")

    if args.dry_run:
        return preview_dry_run(
            args=args,
            model_name=model_name,
            original_model_dir=original_model_dir,
            target_model_dir=target_model_dir,
            backup_path=backup_path,
            repo_url=repo_url,
            token=token,
        )

    try:
        backup_done_path = backup_model_dir(original_model_dir, backup_path, args.overwrite_backup)
        working_model_dir = rename_model_dir(original_model_dir, model_name)
        display_model_dir = working_model_dir
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if not delete_doc_files(working_model_dir, assume_yes=args.yes, dry_run=args.dry_run):
        print_backup_location(backup_done_path, args.dry_run)
        print(f"当前模型目录: {display_model_dir}")
        return 1

    if not cleanup_named_paths(
        model_dir=working_model_dir,
        name=CHECKPOINT_DIR_NAME,
        label=".ipynb_checkpoints 文件夹",
        assume_yes=args.yes,
        dry_run=args.dry_run,
        confirm=True,
        step="4/7",
        stop_message="已取消删除 .ipynb_checkpoints，流程停止，未同步。",
    ):
        print_backup_location(backup_done_path, args.dry_run)
        print(f"当前模型目录: {display_model_dir}")
        return 1

    cleanup_named_paths(
        model_dir=working_model_dir,
        name=UPLOAD_CACHE_NAME,
        label=".ms_upload_cache",
        assume_yes=True,
        dry_run=args.dry_run,
        confirm=False,
        step="4/7",
    )

    if args.sync_mode == "git":
        sync_code = run_git_sync(
            repo_url=repo_url,
            model_dir=working_model_dir,
            token=token,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
            keep_git_workdir=args.keep_git_workdir,
        )
    else:
        sync_code = run_modelscope_upload(
            namespace=args.namespace,
            model_name=model_name,
            model_dir=display_model_dir,
            token=token,
            dry_run=args.dry_run,
        )

    if sync_code != 0:
        print("同步未成功，已停止生成 download.sh 和删除大文件。", file=sys.stderr)
        print_backup_location(backup_done_path, args.dry_run)
        print(f"当前模型目录: {display_model_dir}")
        return sync_code

    cleanup_named_paths(
        model_dir=working_model_dir,
        name=UPLOAD_CACHE_NAME,
        label=".ms_upload_cache",
        assume_yes=True,
        dry_run=args.dry_run,
        confirm=False,
        step="6/7",
    )

    large_files = find_large_files(working_model_dir)
    download_script_path = generate_download_script(
        model_dir=working_model_dir,
        namespace=args.namespace,
        model_name=model_name,
        large_files=large_files,
        dry_run=args.dry_run,
        relative_root=working_model_dir,
    )
    delete_large_files(
        model_dir=working_model_dir,
        large_files=large_files,
        assume_yes=args.yes,
        remove_empty=args.remove_empty_dirs,
        dry_run=args.dry_run,
        relative_root=working_model_dir,
    )

    print("\n处理完成。")
    print_backup_location(backup_done_path, args.dry_run, prefix="  ")
    print(f"  当前模型目录: {display_model_dir}")
    if download_script_path:
        print(f"  下载脚本: {download_script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
