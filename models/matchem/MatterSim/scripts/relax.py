import argparse

from ase.build import bulk

from onescience.utils.mattersim import relax


parser = argparse.ArgumentParser(description="MatterSim structure relaxation")
parser.add_argument(
    "--checkpoint",
    default="../Mattersim/weight/mattersim-v1.0.0-1M.pth",
    help="Path to MatterSim checkpoint",
)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()

converged, atoms = relax(
    bulk("Si", "diamond", a=5.43),
    checkpoint=args.checkpoint,
    filter="FrechetCellFilter",
    fmax=0.01,
    steps=20,
    device=args.device,
)
print(f"Converged: {converged}")
print(f"Energy: {atoms.get_potential_energy():.6f} eV")
