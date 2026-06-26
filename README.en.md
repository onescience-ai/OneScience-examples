# OneScience Examples

Welcome to the OneScience Examples repository! This repository collects example code, training scripts, and inference workflows for a variety of cutting-edge AI/ML models, covering domains such as protein structure prediction, molecular dynamics, computational fluid dynamics, and weather forecasting.

## Project Overview

OneScience Examples is an official model example repository maintained by OneScience, designed to provide researchers and developers with out-of-the-box AI/ML model solutions. Each sub-project includes complete environment configuration, data preparation scripts, training/inference code, and detailed usage documentation.

## Available Models

This repository currently supports the following model categories:

### 🧬 Biological Sciences

| Model | Description |
|-------|-------------|
| [AlphaFold3](./models/AlphaFold3/README.md) | DeepMind's third-generation protein structure prediction model |
| [AlphaGenome](./models/alphagenome/README.md) | DNA sequence analysis and variant scoring model |
| [ESM](./models/ESM/README.md) | ESMFold protein structure prediction |
| [Evo2](./models/evo2/README.md) | Large-scale genomic foundation model |
| [MatRIS](./models/MatRIS/README.md) | Material discovery and structure prediction |
| [OpenFold](./models/OpenFold/README.md) | Open-source protein structure prediction |
| [PINNsformer](./models/PINNsformer/README.md) | Physics-informed neural network |
| [Protenix](./models/protenix/README.md) | Protein structure prediction model |
| [ProteinMPNN](./models/ProteinMPNN/README.md) | Protein sequence design |
| [RFdiffusion](./models/RFdiffusion/README.md) | Protein inverse folding diffusion model |
| [SimpleFold](./models/SimpleFold/README.md) | Lightweight protein structure prediction |
| [UMA](./models/UMA/README.md) | Unified molecular architecture |

### 🧪 Molecular Dynamics

| Model | Description |
|-------|-------------|
| [BENO](./models/BENO/README.md) | Molecular dynamics model |
| [DeepMD](./models/DeepMD/README.md) | Deep potential molecular dynamics |
| [MACE](./models/MACE/README.md) | Interatomic potential model |
| [NEP](./models/NEP/README.md) | Neural network atomic potential |

### 🌤️ Weather Forecasting and Climate

| Model | Description |
|-------|-------------|
| [FourCastNet](./models/FourCastNet/README.md) | Image-based weather forecasting model |
| [FuXi](./models/FuXi/README.md) | Meteorological forecasting model |
| [FengWu](./models/FengWu/README.md) | Weather prediction model |
| [GraphCast](./models/GraphCast/README.md) | Graph neural network for weather forecasting |
| [Pangu-Weather](./models/Pangu_Weather/README.md) | Pangu Weather Large Model |
| [XiHe](./models/XiHe/README.md) | Meteorological prediction model |

### 💧 Computational Fluid Dynamics (CFD)

| Model | Description |
|-------|-------------|
| [CFDBench](./models/CFDBench/README.md) | CFD benchmark dataset |
| [DeepCFD](./models/DeepCFD/README.md) | Deep learning-based CFD model |
| [EagleMeshTransformer](./models/EagleMeshTransformer/README.md) | Mesh graph neural network |
| [GP_for_TO](./models/GP_for_TO/README.md) | Gaussian process optimization |
| [LagrangianMGN](./models/LagrangianMGN/README.md) | Lagrangian graph network |
| [MeshGraphNet](./models/MeshGraphNet/README.md) | Mesh graph neural network |

### 🎨 Design and Generation

| Model | Description |
|-------|-------------|
| [Transolver-Airfoil-Design](./models/Transolver-Airfoil-Design/README.md) | Airfoil design |
| [Transolver-Car-Design](./models/Transolver-Car-Design/README.md) | Car design |

### 📐 Partial Differential Equation Neural Networks (PDENN)

| Model | Description |
|-------|-------------|
| [DeepONet](./models/PDENNEval/DeepONet/README.md) | Deep operator network |
| [FNO](./models/PDENNEval/FNO/README.md) | Fourier operator network |
| [MPNN](./models/PDENNEval/MPNN/README.md) | Message passing neural network |
| [PINN](./models/PDENNEval/PINN/README.md) | Physics-informed neural network |
| [PINO](./models/PDENNEval/PINO/README.md) | Physics-informed operator network |
| [UNO](./models/PDENNEval/UNO/README.md) | Unified operator network |
| [U-Net](./models/PDENNEval/UNet/README.md) | U-Net |
| [WAN](./models/PDENNEval/WAN/README.md) | Wavelet-adaptive network |

## Quick Start

### 1. Environment Setup

Each model project has specific environment requirements. Please refer to the README.md in each model directory for detailed installation instructions.

General dependencies:
```bash
# Base environment (additional configuration may be required depending on the model)
conda create -n onescience python=3.10
conda activate onescience
pip install torch torchvision
```

### 2. Download Models and Data

Most models require downloading pre-trained weights and datasets. Use the download scripts in each model directory:

```bash
cd models/<MODEL_NAME>
bash download.sh
```

### 3. Run Examples

Refer to the execution workflow in each model’s README.md. The general process includes:

1. **Environment Check** – Verify the execution environment
2. **Data Preparation** – Download and extract datasets
3. **Run Inference** – Execute predictions
4. **Validate Output** – Check results

## Project Structure

```
onescience-examples/
├── datasets/              # Dataset-related documentation
├── models/                # Model code
│   ├── AlphaFold3/        # Protein structure prediction
│   ├── FourCastNet/       # Weather forecasting
│   ├── DeepMD/            # Molecular dynamics
│   ├── GraphCast/         # Graph neural network weather
│   ├── PDENNEval/         # PDE networks
│   └── ...                # Other models
└── README.md              # This file
```

## Documentation Guidelines

Each model project includes the following standard documents:

- **README.md** – Project description, installation guide, and usage tutorial
- **manifest.yaml** – Model file manifest
- **conf/** – Configuration files directory
- **scripts/** – Auxiliary scripts directory
- **train.py / inference.py** – Training/inference entry points

## Contribution Guidelines

We welcome issues and pull requests to improve this repository:

1. Submit an Issue to report bugs or propose new model requests
2. Fork this repository
3. Create a new branch for your changes
4. Submit a Pull Request

## License

The code in this repository follows the original licenses of each model project. Please refer to the LICENSE file or license notices in each model’s README.md for specific terms.

## Contact

- Official Website: https://onescience.ai
- Gitee: https://gitee.com/onescience-ai
- GitHub: https://github.com/onescience-ai

## Acknowledgments

Thank you to all open-source model authors and the OneScience team for their contributions.