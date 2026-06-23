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

## 3. Repository Structure

```
mask_interpreter/
├── README.md
├── pyproject.toml              # Package dependencies and configuration
├── example.ipynb               # Quick start tutorial notebook
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
DATA_MODELS_PATH = '/path/to/your/data'
DATA_PATH = '/path/to/your/data/train_test_list'
MODELS_PATH = '/path/to/your/models'
REPO_LOCAL_PATH = 'current working dir'
EXAMPLE_DATA_PATH = '/path/to/example_data/'
```

**Note**: Those variables must be set **before** importing any project modules.

## 5. [Quick Start](quickstart.md)

## Full Data Download

### Allen Cell Collection Dataset

The project uses the Allen Cell Collection dataset from AWS S3. To download and prepare the full field-of-view (FOV) dataset:

**Step 1: Download images**
```bash
cd create_data
python download_and_create_dataset_full.py
```

**Step 2: Create metadata**

After downloading the images, run the metadata creation script:
```bash
python create_metadata.py
```

This second step processes the downloaded images and generates the metadata CSV files required for training.

### Configuration Parameters

Edit the script to customize the download. Key parameters in `download_and_create_dataset_full.py`:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `num_threads` | Number of parallel download threads | `4` |
| `storage_root` | Main directory to save downloaded data | `"/groups/assafza_group/assafza/full_cells_fovs/"` |
| `temp_storage_root` | Temporary directory for processing (use SSD for speed) | `"/scratch/.../full_cells_fovs/"` |
| `num_of_images_per_organelle` | Maximum images to download per organelle | `200` |
| `resacle_z` | Z-axis rescaling factor | `3` |
| `only_csvs` | If `True`, creates metadata CSVs only (no images) | `True` |
| `override` | If `True`, re-downloads existing images | `False` |
| `organelles` | Dictionary of organelles to download | See below |

### Available Organelles

The script supports downloading the following organelles (uncomment in the `organelles` dictionary):

```python
organelles = {
    "Golgi": [],
    "Microtubules": [],
    "Nuclear-envelope": [],
    "Actin-filaments": [],
    "Mitochondria": [],
    "Endoplasmic-reticulum": [],
    "Nucleolus-(Granular-Component)": [],
    "Actomyosin-bundles": [],
    "Plasma-membrane": [],
}
```

### Output Structure

Each downloaded image is a multi-channel TIFF with:
- **Channel 0**: Bright-field image (ROI)
- **Channel 1**: DNA fluorescence (raw ROI)
- **Channel 2**: Membrane fluorescence (raw ROI)
- **Channel 3**: Structure/organelle fluorescence (raw ROI)
- **Channel 4**: DNA segmentation (ROI)
- **Channel 5**: Membrane segmentation (ROI)
- **Channel 6**: Structure/organelle segmentation (ROI)

The script also generates a metadata CSV for each organelle with image paths and metadata.

### Example Usage

To download 50 images each of Mitochondria and Golgi:

```python
# Edit download_and_create_dataset_full.py
num_of_images_per_organelle = 50
only_csvs = False  # Actually download images
organelles = {
    "Mitochondria": [],
    "Golgi": []
}
```

Then run:
```bash
python download_and_create_dataset_full.py
```

## Usage

### Model Types and Differences

MaskInterpreter supports three types of predictive models:

| Model Type | Use Case | Output | Loss Function | Example Application |
|------------|----------|--------|---------------|---------------------|
| **Image-to-Image** | Pixel-wise prediction | 2D/3D image | MSE + L1 mask + PCC target | Organelle prediction, segmentation |
| **Regression** | Continuous value prediction | Scalar(s) | MSE + L1 mask + PCC target | Cell cycle markers, protein levels |
| **Classification** | Category prediction | Class probabilities | MSE + L1 mask + PCC target | CIFAR-10

**Key Differences:**
- **Image-to-Image**: Preserves spatial predictions (e.g., fluorescent organelle images from brightfield)
- **Regression**: Explains scalar outputs (e.g., predicting Cdt1/Geminin marker intensities)
- **Classification**: Identifies regions affecting class probability distributions

All variants share the same core principle: minimize the mask while maintaining high correlation between predictions on original vs. adapted (masked + noise) inputs.

### Training: Image-to-Image Models

For image-to-image models (e.g., organelle prediction, segmentation):

```python
from models.MaskInterpreter import MaskInterpreter
from models.UNETO import get_unet
import tensorflow as tf

# Load your pre-trained predictor
predictor = tf.keras.models.load_model('your_model.h5')
predictor.trainable = False

# Create the mask generator (U-Net based)
adaptor = get_unet(<patch_size,num_filters>, activation="sigmoid")

# Initialize MaskInterpreter
mask_interpreter = MaskInterpreter(
    patch_size=<patch_size>,
    adaptor=adaptor,
    unet=predictor,
    pcc_target=0.95  # Target 95% correlation
)

# Compile with loss weights
mask_interpreter.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    similarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    noise_scale=1.5, ## Noise that remove most of the signal. check figures/1_choose_noise_scale.py
    target_loss_weight=6
)

# Train
mask_interpreter.fit(
    x_train,
    epochs=200,
    batch_size=128,
    validation_data=(x_val, None)
)
```

### Training: Regression Models

For regression models that predict continuous values (e.g., cell cycle markers, protein concentrations):

```python
from models.MaskInterpreterRegression import MaskInterpreterRegression
from models.UNETO import get_unet
import tensorflow as tf

# Load your pre-trained regressor
regressor = tf.keras.models.load_model('cellcycle_marker1.h5')
regressor.trainable = False

# Create the mask generator
# Adaptor takes 2 channels: image + gradient magnitude
adaptor = get_unet((64, 64, 2), activation="sigmoid")

# Initialize MaskInterpreter for Regression
mask_interpreter = MaskInterpreterRegression(
    patch_size=(64, 64, 1),  # Original image size
    adaptor=adaptor,
    regressor=regressor,
    pcc_target=0.95
)

# Compile
mask_interpreter.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    similarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    noise_scale=0.5,
    target_loss_weight=1.75
)

# Build the model
mask_interpreter(np.random.random((1, 64, 64, 1)).astype(np.float32))

# Train
mask_interpreter.fit(
    x_train,
    epochs=200,
    batch_size=128,
    validation_data=(x_val, None)
)
```

**Regression-specific notes:**
- Input augmentation includes gradient magnitude channel (adaptor input shape: [H, W, 2])
- Mask preserves regions important for predicting scalar outputs
- Example: Cell cycle marker prediction (Cdt1/Geminin intensities from brightfield images)

### Training: Classification Models

For classification models (e.g., CIFAR-10, cell type classification):

```python
from models.MaskInterpreterCLF import MaskInterpreter
from models.UNETO import get_unet
import tensorflow as tf

# Load your pre-trained classifier
classifier = tf.keras.models.load_model('cifar10_classifier.h5')
classifier.trainable = False

# Create the mask generator
# Adaptor takes augmented input (32 channels after preprocessing)
adaptor = get_unet((32, 32, 32), activation="sigmoid")

# Initialize MaskInterpreter for Classification
mask_interpreter = MaskInterpreter(
    patch_size=(32, 32, 3),  # Original CIFAR-10 image size
    adaptor=adaptor,
    classifier=classifier,
    weighted_pcc=False,
    pcc_target=0.95
)

# Compile
mask_interpreter.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    similiarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    noise_scale=0.5,
    target_loss_weight=1.75
)

# Build the model
mask_interpreter(np.random.random((1, 32, 32, 3)))

# Train
mask_interpreter.fit(
    x_train,
    epochs=200,
    batch_size=128,
    validation_data=(x_val, None)
)
```

**Classification-specific notes:**
- Input augmentation includes gradient of max predicted probability (computed internally)
- Mask identifies regions affecting class probability distributions
- Works with multi-class outputs (softmax probabilities)
- Example: CIFAR-10 images (32×32×3 RGB images, 10 classes)

### Generating Importance Masks

```python
# Generate mask for a single image
mask = mask_interpreter(image[np.newaxis, ...])

# Visualize
import matplotlib.pyplot as plt
plt.imshow(image[:, :, 0], cmap='gray')
plt.imshow(mask[0, :, :, 0], cmap='jet', alpha=0.5)
plt.title("Importance Mask Overlay")
plt.show()
```

### Evaluating Mask Efficacy

```python
from utils.metrics import tf_pearson_corr

# Original prediction
pred_orig = predictor(image[np.newaxis, ...])

# Create adapted image (important regions preserved, rest is noise)
noise = tf.random.normal(image.shape, stddev=noise_scale)
adapted = mask * image + (1 - mask) * noise

# Prediction on adapted image
pred_adapted = predictor(adapted[np.newaxis, ...])

# Mask efficacy = PCC between predictions
efficacy = tf_pearson_corr(pred_orig, pred_adapted)
print(f"Mask Efficacy (PCC): {efficacy:.4f}")
```


## Experiments

### Organelle Prediction (Label-Free Microscopy)

MaskInterpreter was validated on predicting fluorescent organelle labels from bright-field microscopy images using the Allen Cell Collection dataset.

**Supported Organelles:**
- Mitochondria
- Nuclear Envelope  
- Golgi Apparatus
- Endoplasmic Reticulum
- Microtubules
- Actin Filaments
- Plasma Membrane
- And more...

### Cell Cycle Marker Prediction

Extended to predict cell cycle markers (Cdt1, Geminin) from bright-field images:

```bash
# Train cell cycle regressor
python models/regressor_cellcycle.py

# Train MaskInterpreter for cell cycle
python models/MaskInterpreterRegression.py

# Evaluate and generate visualizations
python models/mi_reg_cellcycle_eval.py
```

### CIFAR-10 Classification

Demonstrated on image classification to show generality:

```bash
# Train cifar10 classifier
python clf-cifar10.py
```
```bash
# Train MaskInterpreter for cifar10 and evaluate
python models/MaskInterpreterCLF.py
```

## Cell cycle and CIFAR10 Results

<p align="center">
  <img src="figures/results_comparison.png" alt="Results" width="800"/>
</p>


## Recreate Figures from Paper

To reproduce all figures from the paper, run the scripts in the `figures/` folder in numerical order. 

**Important**: Before running, update the following paths in each script:
- Model checkpoint paths (trained MaskInterpreter and predictor models)
- Data directory paths (Allen Cell Collection dataset location)
- Output directory paths for generated figures

### Running Scripts in Order

```bash
cd figures

# 0. Reproduce U-Net scores
python 0_reproduce_unet_scores.py

# 1. Choose noise scale parameter
python 1_choose_noise_scale.py

# 2. Choose threshold for binarization
python 2_choose_th.py

# 3. Calculate U-Net prediction scores
python 3_calculate_unet_scores.py

# 4. Calculate explanation mask efficacy
python 4_calculate_explanation_mask_efficacy.py

...
```

**Note**: Some scripts may have dependencies on outputs from previous scripts. Run them in the order listed above to ensure all required intermediate files are generated.

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
