# Usage

## Model Types and Differences

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

## Training: Image-to-Image Models

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

## Training: Regression Models

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

## Training: Classification Models

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

## Generating Importance Masks

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

## Evaluating Mask Efficacy

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
