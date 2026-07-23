# -*- coding: utf-8 -*-
"""
下载 predictia/cerra_tas_vqvae 的模型文件到当前目录。
在 **有外网** 的机器上运行(你的本地电脑,或有外网的超算容器):

    pip install huggingface_hub
    python download_weights.py

会把 config.json / diffusion_pytorch_model.bin / README.md 拉到当前目录。
下载完成后,把整个目录(含 diffusion_pytorch_model.bin)上传到
/root/private_data/Cerra_tas_vqvae/ 即可。

如果你无法运行 Python,也可以直接用浏览器下载这一个文件(356 kB):
    https://huggingface.co/predictia/cerra_tas_vqvae/resolve/main/diffusion_pytorch_model.bin
并放到 /root/private_data/Cerra_tas_vqvae/ 目录下。
"""
import os
from huggingface_hub import hf_hub_download

REPO = "predictia/cerra_tas_vqvae"
FILES = ["config.json", "diffusion_pytorch_model.bin", "README.md"]
here = os.path.dirname(os.path.abspath(__file__))

for fn in FILES:
    print(f"downloading {fn} ...")
    p = hf_hub_download(repo_id=REPO, filename=fn, local_dir=here,
                        local_dir_use_symlinks=False)
    print(f"  -> {p}")

print("\n完成。当前目录内容:")
for fn in sorted(os.listdir(here)):
    fp = os.path.join(here, fn)
    if os.path.isfile(fp):
        print(f"  {fn:40s} {os.path.getsize(fp):>10,} bytes")
