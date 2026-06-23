"""
MaskInterpreter for Cell Cycle Marker Regression
=================================================

This script implements a MaskInterpreter model adapted for regression tasks,
specifically designed to generate interpretable importance masks that explain
cell cycle marker predictions from brightfield microscopy images.

Overview:
---------
The MaskInterpreterRegression learns to generate importance masks that identify
which regions of a cell image are critical for predicting cell cycle markers
(Cdt1 and Geminin intensities). By perturbing non-important regions with noise
while preserving important regions, the regressor's output should remain similar.

This is an adaptation of the classification-based MaskInterpreter for regression
tasks on biological microscopy data.

Architecture:
-------------
    1. Input Augmentation:
       - Original grayscale image (64x64x1)
       - Gradient magnitude channel: computes the gradient of the regressor's
         output w.r.t. the input, normalized to [0,1]
       - Augmented input shape: (64x64x2)
    
    2. Generator Network:
       - Adaptor network (U-Net) that takes augmented input
       - Outputs importance mask (64x64x1)
       - Mask values in [0,1] where 1 = important, 0 = not important
    
    3. Regressor:
       - Pretrained cell cycle marker regression model (frozen weights)
       - Separate models for Marker 1 (Cdt1) and Marker 2 (Geminin)
       - Outputs single value in [0,1] representing marker intensity

Training Objective:
-------------------
    The model is trained to minimize a composite loss:
    
    1. Similarity Loss (MSE):
       - MSE between regressor outputs on original vs. adapted images
       - Encourages the mask to preserve regressor-relevant information
    
    2. Mask Sparsity Loss (L1):
       - Mean absolute value of the mask
       - Encourages smaller masks (only highlight truly important regions)
    
    3. PCC Target Loss:
       - |pcc_target - actual_pcc|
       - Encourages predictions to maintain a target Pearson correlation
       - pcc_target=0.95 means we want 95% correlation between outputs
    
    Total Loss = (similarity_loss * sim_weight) + 
                 (mask_loss * mask_weight) + 
                 (pcc_loss * target_weight)

Mask Efficacy:
--------------
    Mask efficacy is measured as the Pearson Correlation Coefficient (PCC)
    between the regressor's predictions on:
    - The original image
    - The adapted image (important regions preserved, others replaced with noise)
    
    Higher PCC = mask successfully identifies important regions for regression
    Lower PCC = mask fails to capture what the regressor uses

Data Format:
------------
    Images (.npy files):
        - Shape: (T, C, Y, X) - T=timepoints, C=channels, Y/X=spatial
        - Loaded as (64, 64, 1) grayscale images
    
    Labels (.npy files):
        - Shape: (C, T, N) - C=markers (2), T=timepoints, N=indices
        - Two markers: Cdt1 (index 0) and Geminin (index 1)

Workflow:
---------
    1. Load cell cycle dataset (train/val/test splits)
    2. Normalize images using per-dataset statistics
    3. Load pretrained regression models for Marker 1 and Marker 2
    4. Create two MaskInterpreterRegression instances (one per marker)
    5. Train both mask interpreters on training data
    6. Generate importance masks that explain each regressor's predictions

Output Files:
-------------
    Models:
        - cellcycle_mi_marker1/: Trained MaskInterpreter weights for Marker 1
        - cellcycle_mi_marker2/: Trained MaskInterpreter weights for Marker 2
    
    Required Input Models:
        - cellcycle_marker1.h5: Pretrained Cdt1 regressor
        - cellcycle_marker2.h5: Pretrained Geminin regressor

Configuration Flags:
--------------------
    load (bool): If True, load pre-trained MaskInterpreter weights.
                 Default: False
    
    train (bool): If True, train the MaskInterpreter models.
                  Default: True
    
    adabatch (bool): If True, apply adaptive batch normalization.
                     Default: False
    
    num_samples (int): Number of samples to visualize.
                       Default: 100
    
    num_samples_subset (int): Number of samples for computing metrics.
                              Default: 500

Key Classes and Functions:
--------------------------
    MaskInterpreterRegression (keras.Model):
        - __init__: Initialize with adaptor network and regressor
        - compile: Set optimizer and loss weights
        - train_step: Custom training loop with gradient augmentation
        - call: Generate importance mask for input images
        - predict_from_noisy: Get regressor prediction on adapted image
        - _augment_input_with_gradients: Add gradient magnitude as extra channel
    
    Data Loading Functions:
        - load_single_sample: Load single image-label pair for parallel processing
        - load_cell_cycle_data: Load all data from directory with multiprocessing

Training Configuration:
-----------------------
    - Optimizer: Adam (learning_rate=5e-4)
    - Loss Weights: similarity=1.0, mask=1.0, target=1.75
    - Noise Scale: 0.5
    - PCC Target: 0.95
    - Batch Size: 128
    - Max Epochs: 200
    - Early Stopping: patience=7, restore_best_weights=True

Dependencies:
-------------
    - tensorflow / keras
    - numpy
    - tqdm
    - multiprocessing
    - Custom modules: metrics (tf_pearson_corr), callbacks, models.UNETO
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.metrics import tf_pearson_corr
from utils.callbacks import *
import os
import glob
import importlib
from models.regressor_cellcycle import get_file_list, create_lazy_dataset, compute_dataset_statistics

# Ensure eager execution (TF2 does this by default)
tf.compat.v1.enable_eager_execution()

from models.regressor_cellcycle import adapt_batch_norm

###############--LOAD--##########################
load = False
train = True
adabatch = True

# Number of samples to visualize
num_samples = 100
num_samples_subset = 500
base_dir = os.path.join(gv.CWD, 'cell_cycle')

class MaskInterpreterRegression(keras.Model):
    def __init__(self, patch_size, adaptor, regressor, pcc_target=0.9, **kwargs):
        """
        Mask Interpreter adapted for regression tasks.
        
        Args:
            patch_size (tuple): Original input image shape, e.g. (64, 64, 1)
            adaptor (keras.Model): A network that, given processed inputs, outputs a mask.
            regressor (keras.Model): A pretrained regression model (e.g., for cell cycle markers).
            pcc_target (float): Target PCC value between the regressor's predictions on
                               the original and perturbed images. Defaults to 0.9.
        """
        super(MaskInterpreterRegression, self).__init__(**kwargs)
        self.pcc_target = pcc_target
        self.regressor = regressor  # this regressor is assumed fixed
        
        # Augment input with gradient: new channel count = original channels + 1.
        augmented_input_shape = (patch_size[0], patch_size[1], patch_size[2] + 1)
        image_input = keras.layers.Input(shape=augmented_input_shape, dtype=tf.float32)
        
        # Pass the augmented input directly to the adaptor
        # The adaptor expects (H, W, C+1) where C=1, so (64, 64, 2)
        importance_mask = tf.cast(adaptor(image_input), dtype=tf.float64)
        self.generator = keras.Model(image_input, importance_mask, name="generator")
        
        # ---- Define metrics ----
        self.similarity_loss_tracker = keras.metrics.Mean(name="similarity_loss")
        self.binary_size_mask = keras.metrics.BinaryAccuracy(name="binary_size_mask")
        self.importance_mask_size = keras.metrics.Mean(name="importance_mask_size")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.pcc = keras.metrics.Mean(name="pcc")
        self.stop = keras.metrics.Mean(name="stop")

    def compile(self, g_optimizer, similarity_loss_weight=1.0, mask_loss_weight=1.0,
                noise_scale=1.0, target_loss_weight=0.0, run_eagerly=False):
        """
        Args:
            g_optimizer: Optimizer for updating the adaptor network.
            similarity_loss_weight (float): Weight for the similarity loss.
            mask_loss_weight (float): Weight for the mask loss.
            noise_scale (float): Standard deviation of the noise applied.
            target_loss_weight (float): Weight for the PCC target loss.
            run_eagerly (bool): Whether to run eagerly.
        """
        super(MaskInterpreterRegression, self).compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.similarity_loss_weight = similarity_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.noise_scale = noise_scale
        self.target_loss_weight = target_loss_weight

    @property
    def metrics(self):
        return [
            self.similarity_loss_tracker,
            self.total_loss_tracker,
            self.binary_size_mask,
            self.importance_mask_size,
            self.pcc,
            self.stop,
        ]
    
    def _augment_input_with_gradients(self, x):
        """
        Compute the gradient of the regressor's output with respect to the input x
        and use its magnitude as an additional channel.
        
        Args:
            x: Tensor of shape (B, H, W, C) (e.g. (B, 64, 64, 1))
        Returns:
            augmented_x: Tensor of shape (B, H, W, C+1)
        """
        x_float = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_float)
            preds = self.regressor(x_float)  # shape (B, 1) for regression
            # Use the regression output directly
            output = tf.reduce_sum(preds, axis=1)  # shape (B,)
        
        # Compute gradients of the output with respect to x
        grad = tape.gradient(output, x_float)  # shape (B, H, W, C)
        
        # Compute the L2 norm of the gradients along the channel dimension
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=-1, keepdims=True))  # shape (B, H, W, 1)
        
        # Normalize grad_norm to [0,1] per sample
        epsilon = 1e-8
        min_val = tf.reduce_min(grad_norm, axis=[1,2,3], keepdims=True)
        max_val = tf.reduce_max(grad_norm, axis=[1,2,3], keepdims=True)
        grad_norm = (grad_norm - min_val) / (max_val - min_val + epsilon)
        
        # Concatenate the original input with the grad_norm as an extra channel
        augmented_x = tf.concat([x, tf.cast(grad_norm, x.dtype)], axis=-1)  # shape (B, H, W, C+1)
        return augmented_x

    def train_step(self, data):
        # Unpack data (images and labels are provided, but we only use images)
        if isinstance(data, tuple):
            x, _ = data  # Unpack images and labels (labels not used in training)
        else:
            x = data
        
        x = tf.cast(x, dtype=tf.float64)  # original input, shape (B, H, W, C)
        
        # Compute regressor output on the original image (for loss computation)
        regressor_target = tf.cast(self.regressor(x), dtype=tf.float64)
        
        # Augment input with gradient channel
        augmented_x = self._augment_input_with_gradients(tf.cast(x, tf.float32))
        augmented_x = tf.cast(augmented_x, tf.float32)
        
        with tf.GradientTape() as tape:
            # Generate importance mask from the augmented input
            importance_mask = self.generator(augmented_x)  # shape (B, H, W, 1)
            
            # Create random noise
            normal_noise = tf.random.normal(tf.shape(importance_mask), stddev=self.noise_scale, dtype=tf.float64)
            
            # Compute adapted (noisy) image: use importance_mask to mix x and noise
            adapted_image = (importance_mask * x) + (normal_noise * (1 - importance_mask))
            
            # Get regressor's output on the adapted image
            regressor_output = tf.cast(self.regressor(tf.cast(adapted_image, tf.float32)), dtype=tf.float64)
            
            # Compute similarity loss (MSE between regressor outputs on original and adapted images)
            similarity_loss = tf.reduce_mean(tf.square(regressor_target - regressor_output))
            
            # Regularization on the mask (encourage sparsity)
            mean_importance_mask = tf.reduce_mean(importance_mask)
            mask_loss = tf.reduce_mean(tf.abs(importance_mask))
            
            # Compute PCC between regressor outputs
            pcc_value = tf_pearson_corr(regressor_target, regressor_output)
            pcc_loss = tf.abs(self.pcc_target - pcc_value)
            
            # Total loss
            total_loss = (similarity_loss * self.similarity_loss_weight) + \
                         (mask_loss * self.mask_loss_weight) + \
                         (pcc_loss * self.target_loss_weight)
        
        grads = tape.gradient(total_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics
        self.similarity_loss_tracker.update_state(similarity_loss)
        self.importance_mask_size.update_state(1 - mean_importance_mask)
        self.binary_size_mask.update_state(tf.zeros_like(importance_mask), importance_mask)
        self.pcc.update_state(pcc_value)
        self.total_loss_tracker.update_state(total_loss)
        self.stop.update_state(pcc_loss + mean_importance_mask)
        
        return {
            "similarity_loss": self.similarity_loss_tracker.result(),
            "binary_size": self.binary_size_mask.result(),
            "importance_mask_size": self.importance_mask_size.result(),
            "total_loss": self.total_loss_tracker.result(),
            "pcc": self.pcc.result(),
            "stop": self.stop.result()
        }
    
    def test_step(self, data):
        return self.train_step(data)
    
    def call(self, inputs):
        # For inference, augment the input with gradient channel and produce the mask
        augmented_x = self._augment_input_with_gradients(tf.cast(inputs, tf.float32))
        return self.generator(tf.cast(augmented_x, tf.float32))
    
    def predict_from_noisy(self, inputs):
        # Given a single image, augment it with gradient channel, compute adapted image and return regressor prediction
        augmented_x = self._augment_input_with_gradients(tf.cast(np.expand_dims(inputs, axis=0), tf.float32))
        mask = self.generator.predict(augmented_x)
        mask = np.squeeze(mask)  # shape becomes (H,W) or (H,W,1)
        
        if mask.ndim == 2:
            mask_expanded = np.expand_dims(mask, axis=-1)
        else:
            mask_expanded = mask
            
        noise = tf.random.normal(shape=inputs.shape, stddev=self.noise_scale, dtype=tf.float64).numpy()
        adapted_image = (mask_expanded * inputs.astype(np.float64)) + ((1 - mask_expanded) * noise)
        pred_value_adapted = self.regressor.predict(np.expand_dims(adapted_image.astype(np.float32), axis=0))
        return pred_value_adapted


if __name__ == "__main__":
    # Load pre-split datasets using lazy loading
    base_data_dir = os.path.join(gv.BASE_PATH, 'path/to/Cell_Cycle_Data')
    
    print("\n" + "="*60)
    print("Preparing Lazy Loading Datasets")
    print("="*60)
    
    print("\nScanning training files...")
    train_files, train_samples = get_file_list(os.path.join(base_data_dir, "train"))
    
    print("\nScanning validation files...")
    val_files, val_samples = get_file_list(os.path.join(base_data_dir, "val"))
    
    print("\nScanning test files...")
    test_files, test_samples = get_file_list(os.path.join(base_data_dir, "test"))
    
    # Compute normalization statistics from training set
    print("\nComputing normalization statistics from training data...")
    train_mean, train_std = compute_dataset_statistics(train_files, num_samples=10000)
    
    # Create lazy loading datasets
    batch_size_mi = 128
    print(f"\nCreating lazy loading datasets with batch_size={batch_size_mi}...")
    train_ds, _ = create_lazy_dataset(train_files, train_mean, train_std, shuffle=True, batch_size=batch_size_mi)
    val_ds, _ = create_lazy_dataset(val_files, train_mean, train_std, shuffle=False, batch_size=batch_size_mi)
    test_ds, _ = create_lazy_dataset(test_files, train_mean, train_std, shuffle=False, batch_size=batch_size_mi)
    
    print(f"\nLazy loading dataset summary:")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")
    print(f"  Test samples: {test_samples}")
    
    ###############--LOAD TRAINED REGRESSORS--##########################
    
    print("\nLoading trained regression models...")
    marker1_model = tf.keras.models.load_model(f'{base_dir}/cellcycle_marker1.h5')
    marker1_model.trainable = False
    print("Marker 1 model loaded (weights frozen)")
    
    marker2_model = tf.keras.models.load_model(f'{base_dir}/cellcycle_marker2.h5')
    marker2_model.trainable = False
    print("Marker 2 model loaded (weights frozen)")
    
    # Apply adaptive batch normalization if enabled
    if adabatch:
        print("\n" + "="*60)
        print("Applying Adaptive Batch Normalization...")
        print("="*60)
        print("Loading adaptation batch from training data...")
        
        # Load a batch for adaptation
        from models.regressor_cellcycle import load_sample_from_file
import global_vars as gv
        adapt_samples = []
        adapt_count = 0
        adapt_batch_size = 1000
        
        for file_info in train_files[:min(20, len(train_files))]:
            num_timepoints = file_info[2]
            for t in range(num_timepoints):
                if adapt_count >= adapt_batch_size:
                    break
                bf, _ = load_sample_from_file(file_info, t)
                bf_norm = (bf - train_mean) / train_std
                adapt_samples.append(bf_norm)
                adapt_count += 1
            if adapt_count >= adapt_batch_size:
                break
        
        adapt_data = np.array(adapt_samples)
        print(f"Adapting batch normalization on {len(adapt_data)} samples...")
        adapt_batch_norm(marker1_model, adapt_data, batch_size=32)
        adapt_batch_norm(marker2_model, adapt_data, batch_size=32)
        print("✓ Batch normalization adaptation complete")
        del adapt_data, adapt_samples
    
    ###############--DEFINE ADAPTOR--##########################
    
    from models.UNETO import *
    # Adaptor for 64x64 grayscale images with 2 channels (image + gradient)
    adaptor_m1 = get_unet((64, 64, 2), activation="sigmoid")
    adaptor_m1.summary()
    
    adaptor_m2 = get_unet((64, 64, 2), activation="sigmoid")
    
    ###############--CREATE MASK INTERPRETERS--##########################
    
    print("\n" + "="*60)
    print("Creating Mask Interpreter for Marker 1...")
    print("="*60)
    
    mi_marker1 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m1,
        regressor=marker1_model,
        pcc_target=0.95
    )
    
    mi_marker1.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=2.5
    )
    
    # Build the model
    mi_marker1(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    print("\n" + "="*60)
    print("Creating Mask Interpreter for Marker 2...")
    print("="*60)
    
    mi_marker2 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m2,
        regressor=marker2_model,
        pcc_target=0.95
    )
    
    mi_marker2.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=2.5
    )
    
    # Build the model
    mi_marker2(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    ###############--SETUP CALLBACKS--##########################
    
    model_path_m1 = f"{base_dir}/cellcycle_mi_marker1"
    model_path_m2 = f"{base_dir}/cellcycle_mi_marker2"
    
    checkpoint_callback_m1 = SaveModelCallback(
        1, mi_marker1, model_path_m1, 
        monitor="val_stop", term="val_pcc", term_value=1.0
    )
    
    checkpoint_callback_m2 = SaveModelCallback(
        1, mi_marker2, model_path_m2,
        monitor="val_stop", term="val_pcc", term_value=1.0
    )
    
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_stop',
        patience=7,
        restore_best_weights=True
    )
    
    ###############--TRAINING--##########################
    
    if load:
        print("\nLoading pre-trained Mask Interpreters...")
        mi_marker1.load_weights(model_path_m1)
        mi_marker2.load_weights(model_path_m2)
    
    if train:
        print("\n" + "="*60)
        print("Training Mask Interpreter for Marker 1...")
        print("="*60)
        
        mi_marker1.fit(
            train_ds,
            epochs=200,
            steps_per_epoch=train_samples // batch_size_mi,
            callbacks=[early_stop_callback, checkpoint_callback_m1],
            validation_data=val_ds,
            validation_steps=val_samples // batch_size_mi
        )
        
        print("\n" + "="*60)
        print("Training Mask Interpreter for Marker 2...")
        print("="*60)
        
        mi_marker2.fit(
            train_ds,
            epochs=200,
            steps_per_epoch=train_samples // batch_size_mi,
            callbacks=[early_stop_callback, checkpoint_callback_m2],
            validation_data=val_ds,
            validation_steps=val_samples // batch_size_mi
        )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
