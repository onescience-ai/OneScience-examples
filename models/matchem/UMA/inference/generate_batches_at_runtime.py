import os

# 自动定位 UMA 旋转基文件 Jd.pt（如果在仓库根目录 weight/ 下存在）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JD_PATH = os.path.join(_REPO_ROOT, "weight", "Jd.pt")
if os.path.isfile(_JD_PATH):
    os.environ.setdefault("ONESCIENCE_UMA_JD_PATH", _JD_PATH)

from ase.build import bulk

from onescience.datapipes.materials.custom_stack.core.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from onescience.utils.uma.units.mlip_unit import load_predict_unit

# 构建多个结构，可替换为 molecule() 或 slab(...)
atoms_list = [
    bulk("Pt"),
    bulk("Cu"),
    bulk("NaCl", crystalstructure="rocksalt", a=2.0),
]

# 转换为 AtomicData 并赋予任务名
atomic_data_list = [
    AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
]

# 合并成一个 batch
batch = atomicdata_list_to_batch(atomic_data_list)

# 加载模型（默认从仓库根目录的 weight/ 下读取）
checkpoint_path = os.environ.get(
    "UMA_CHECKPOINT_PATH",
    os.path.join(_REPO_ROOT, "weight", "uma-s-1p1_converted.pt"),
)
predictor = load_predict_unit(checkpoint_path, device="cuda")

# 执行推理
preds = predictor.predict(batch)

# 输出每个结构的能量和原子力
for i, atoms in enumerate(atoms_list):
    energy = preds["energy"][i].item()
    forces = preds["forces"][batch.batch == i].cpu().numpy()

    print(f"\nStructure #{i + 1}: {atoms.get_chemical_formula()}")
    print("Predicted energy:", energy)
    print("Predicted forces:\n", forces)
