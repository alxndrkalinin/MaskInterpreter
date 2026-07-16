# MaskInterpreter | Trustworthy in silico labeling via semantic visual interpretability of image-to-image translation
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Lion Ben Nedava<sup>1*</sup>, Gad Miller<sup>1*</sup>, Nitsan Elmalam<sup>1</sup>, Mateheus Viana<sup>3</sup>, Jianxu Chen<sup>2</sup>, Nathalie Gaudreault<sup>3</sup>, Sussane Rafelski<sup>3</sup>, Assaf Zaritsky<sup>1</sup>

\*Equal contribution

1. Institute for Interdisciplinary Computational Science, Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Beer-Sheva 84105, Israel
2. Leibniz-Institut fur Analytische Wissenschaften - ISADS - e.V, Dortmund, Germany
3. Allen Institute for Cell Science, Seattle, WA, USA

## 1. Abstract

Cross-modality image translation promises to provide multiple layers of biological information from a single input, yet its practical application is stalled by a lack of interpretability and the inability to account for model imperfections. In silico labeling, the inference of organelle localization from label-free images, is a primary example where this black-box nature limits adoption. We present Mask Interpreter, a generalized method for semantic visual interpretability of image-to-image translation models. By uncovering organelle-specific "explanation signatures", we demonstrate that models leverage unique and reproducible biological patterns. Mask Interpreter outperforms traditional xAI approaches, identifies batch effects and localized prediction errors when ground-truth fluorescence is unavailable. Our supervised confidence modeling provides fine-grained reliability assessment at single-cell resolution, enabling the automated exclusion of artifacts from downstream analyses. By bridging the gap between computational inference and meaningful biological features, Mask Interpreter transforms in silico labeling into a rigorous, evidence-based instrument for scientific discovery across diverse biomedical imaging modalities.

<p align="center">
  <img src="figures/figure1.png" alt="MaskInterpreter Architecture" height="600"/>
</p>

**Figure 1. Interpreting in silico labeling using Mask Interpreter**. (A) Training of in silico labeling models using matched label-free and fluorescence images. (B) Example of predictions with/without using MaskInterpreter’s importance mask. (C-F) Training and inference pipeline schematic. 

See Paper (link) for details.

## 2. Overview
Deep learning models often operate as "black boxes," making it difficult to understand which input features drive their predictions. MaskInterpreter addresses this by learning a per-organelle mask generator network that identifies important regions through a novel training objective:

- **Preserve predictions**: Important regions (high mask values) should be sufficient to maintain the model's original prediction
- **Minimize mask size**: The mask should be as minimal as possible, highlighting only truly essential regions
- **Target correlation**: Predictions on masked inputs should maintain a specified correlation with the original predictions

### Key Features
- **Self-supervised training** - No ground truth explanations needed
- **Model-agnostic** - Works with any differentiable predictor (e.g. classifiers, regressors, image-to-image models)
- **New measurement to quantify explanations** - the Pearson correlation coefficient (PCC) between the predictions derived from the unperturbed input, and the predictions derived from the importance mask-induced noisy inputs

## 2b. PyTorch port (`mask_interpreter/`)

The method has been ported to **PyTorch** in the `mask_interpreter/` package (NCHW/NCDHW,
float32). It is **model-agnostic**: you bring your own frozen `nn.Module` predictor — no
TF weights or `cell_imaging_utils` needed. The original TensorFlow modules (`models/`,
`dataset.py`, `mg_analyzer.py`, `utils/`, `gui/`, `figures/`) are **retained on disk for
reference** and are not required by the PyTorch package.

### Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e .            # torch stack from pyproject.toml
python -m pytest tests/        # 41 integration tests
```

### The three variants

| variant | class | predictor output | front-end | PCC clamp |
|---|---|---|---|---|
| image-to-image (2D/3D) | `Image2ImageInterpreter` | image | 2× learned conv on input & prediction | one-sided (stops at target) |
| classification (2D) | `ClassificationInterpreter` | class scores | 1× learned Conv2d on grad-augmented input | two-sided |
| regression (2D) | `RegressionInterpreter` | scalar(s) | none (adaptor on grad-augmented input) | two-sided |

The mask is a sigmoid U-Net (`unet.UNet`, dynamic depth driven by the first spatial
axis). The adapted image is `mask*x + noise*(1-mask)`; the loss is
`w_sim*MSE + w_mask*size + w_pcc*pcc`. Per-variant seed hyperparameters live in
`config.py` (`IMAGE2IMAGE_LOSS`, `CLASSIFICATION_LOSS`, `REGRESSION_LOSS`).

### Train (image-to-image)

```python
import torch
from mask_interpreter import Image2ImageInterpreter, freeze
from mask_interpreter.train import Trainer

predictor = freeze(my_pretrained_unet)                 # any nn.Module: x -> prediction
model = Image2ImageInterpreter(predictor, spatial_size=(32, 128, 128), ndim=3)

trainer = Trainer(model, lr=1e-4, checkpoint_path="mg.pt",
                  monitor="val_stop", term="val_pcc", term_value=0.03)
trainer.fit(train_loader, val_loader, epochs=100)      # batches yield (x[, seg])

model.eval()
with torch.no_grad():
    mask = model(x)                                    # importance mask in [0, 1]
```

Classification/regression are analogous; their batches are `x` (or `(x, y)`), and the
gradient-augmentation channel is computed internally.

### Evaluate (FOV threshold sweep)

```python
from mask_interpreter.analyze import Analyzer

az = Analyzer(model, "image_list.csv", input_col="channel_signal",
              target_col="channel_dna", patch_size=(32, 128, 128, 1))
az.calc_unet_pcc("out/", images=range(10))                    # predictor quality vs target
az.find_noise_scale("out/", images=range(10))                 # noise-std selection
az.analyze_th("out/", mode="agg", images=range(10), save_image=True)  # mask-size / context sweep
```

Outputs: `pcc_results.csv`, `mask_size_results.csv`, `context_results.csv`, and per-image
tiffs (`input_/target_/unet_prediction_/mask_/noisy_*`). Noise is drawn **once per image**
and reused across thresholds (a faithful, reproducible sweep).

### What changed vs. the TF version

- `cell_imaging_utils` I/O → `tifffile` + `pandas`; `keras.Model` custom `train_step` →
  `nn.Module` + plain training loop; float64 → float32.
- PCC is consistent with [`cubic`](https://github.com/alxndrkalinin/cubic): the numpy
  eval path calls `cubic.metrics.pcc`; the training loss mirrors its math in a
  differentiable torch form (`cubic.pcc` returns a float and detaches, so it can't carry
  the loss gradient). NaN-on-zero-variance and `[-1,1]` clipping match cubic.
- Silent failure modes removed: the analyzer no longer swallows errors into `-1`, and
  `predict` raises a real error on CUDA OOM (after retrying at smaller batch sizes).
- `global_vars.py` (mutated at import) → `config.py` dataclasses (`Config`, `*Loss`).

See `PYTORCH_PORT_PLAN.md` for the full design and fidelity notes.

## 3. Repository Structure

```
mask_interpreter/
├── README.md
├── pyproject.toml              # Package dependencies and configuration
├── example.ipynb               # Quick start tutorial notebook
│
├── md/                         # Documentation files
│   ├── quickstart.md           # Quick start guide
│   ├── data.md                 # Full data download instructions
│   ├── usage.md                # Usage examples and training
│   ├── examples.md             # Additional examples
│   └── reproduce.md            # Reproducing paper figures
│
├── models/                     # Core MaskInterpreter implementations
│   ├── MaskInterpreter.py      # Image-to-image models
│   ├── MaskInterpreterRegression.py   # Regression models
│   ├── MaskInterpreterCLF.py   # Classification models
│   ├── regressor_cellcycle.py  # Cell cycle marker regression
│   ├── clf-cifar10.py          # CIFAR-10 classifier
│   └── UNETO.py                # U-Net architecture for mask generation
│
├── create_data/                # Data download and preparation scripts
│   ├── download_and_create_dataset_full.py
│   ├── download_and_create_dataset_singlecell.py
│   ├── create_metadata.py
│   └── segment_and_create_pertrub_dataset.py
│
├── figures/                    # Scripts to reproduce paper figures
│   ├── 0_reproduce_unet_scores.py
│   ├── 1_choose_noise_scale.py
│   ├── 2_choose_th.py
│   ├── 3_calculate_unet_scores.py
│   ├── 4_calculate_explanation_mask_efficacy.py
│   └── ...
│
├── gui/                        # Graphical user interface
│   ├── gui.py
│   └── gui_logic.py
│
├── utils/                      # Utility modules
│   ├── callbacks.py            # Training callbacks
│   ├── metrics.py              # Evaluation metrics (PCC, etc.)
│   └── utils.py                # Helper functions
│
├── dataset.py                  # Data loading utilities
├── global_vars.py              # Global configuration and paths
├── mg_analyzer.py              # Mask generator analyzer
├── train.py                    # Main training script
└── test.py                     # Testing utilities
```

## 4. General Installation and setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda (recommended for environment management)

### Download Trained Models and Example Data

Pre-trained models and example data are available from Zenodo for quick start and reproducibility.

#### Download from Zenodo

Visit the Zenodo repository to download the required files:

**Zenodo Link:** [https://zenodo.org/records/18590674](https://zenodo.org/records/18590674)

The archive contains:
- **Pre-trained in silico labeling models** - Trained on various organelles
- **Pre-trained MaskInterpreter models** - Corresponding interpretation models for each predictor
- **Example data** - Sample images with metadata for testing and validation
- **Train and test lists of the full data**

#### Extraction and Setup

After downloading, extract the contents and set the paths accordingly:

```bash
# Download the archive from Zenodo
wget https://zenodo.org/records/18590674/files/models_and_data.zip

# Extract to your desired location
unzip models_and_data.zip -d /path/to/your/directory

# The extracted files under models_and_data dir will contain:
# - example_data (example dataset with train/test CSV files)
# - models/ (pre-trained models)
# - train_test_list/ (train and test lists of the full data)
```

Make sure to update your environment variables (see next section) to point to these directories.

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/lionben89/cell_generator.git
cd cell_generator
```

2. **Create conda environment**
```bash
conda create -n maskinterpreter python=3.9 tensorflow-gpu=2.6
conda activate maskinterpreter
```

3. **Install the package in editable mode**
```bash
pip install -e .
```

This will install all dependencies from `pyproject.toml` automatically


### Verify Installation

**Note**: Run this verification on the terminal of the computer/node with GPU access.

```bash
# Activate the environment
conda activate maskinterpreter

# Run verification script
python -c "
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
print('GPUs Available:', tf.config.list_physical_devices('GPU'))

from models.MaskInterpreter import MaskInterpreter
print('MaskInterpreter imported successfully!')
"
```

### Configuration

The project uses `global_vars.py` for configuration including paths for data, models, and the repository. You can configure these either by:

Open [global_vars.py](global_vars.py) and update the path variables at the top of the file:

```python
# ============================================
# Path Configuration
# ============================================
# Base paths - update these to match your environment
BASE_PATH = '/path/to/your/data'
DATA_PATH = '/path/to/your/data/train_test_list'
CWD = 'current working dir'
```

**Note**: Those variables must be set **before** importing any project modules.

## 5. [Quick Start](md/quickstart.md)

## 6. [Full Data Download](md/data.md)

## 7. [Usage](md/usage.md)

## 8. [Additional Examples](md/examples.md)

## 9. [Recreate Figures from Paper](md/reproduce.md)

## PyTorch Implementation
> **PyTorch Implementation and implementation of supervised confidence model**: For a PyTorch version of MaskInterpreter and tools for assessing the supervised prediction quality at inference time using MaskInterpreter, see the companion repository: [https://github.com/zaritskylab/Interpretability](https://github.com/zaritskylab/Interpretability)

## Citation

If you use MaskInterpreter in your research, please cite:

```bibtex
@article{TODO,
  title={TODO},
  author={Ben Nedava, Lion and Miller, Gad and Zaritsky, Assaf},
  journal={bioRxiv},
  year={2026}
}
```

## Acknowledgments

- [Allen Institute for Cell Science](https://alleninstitute.org/cell-science) for the cell imaging datasets and review of the paper.

## Contact

- **Email**: assafzar@gmail.com , lionben89@gmail.com, gadmicha@post.bgu.ac.il
- **Lab**: [Zaritsky Lab](https://www.https://www.assafzaritsky.com/)
