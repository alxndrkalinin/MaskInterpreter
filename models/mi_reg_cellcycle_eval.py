"""
MaskInterpreter Evaluation Script for Cell Cycle Marker Regression
===================================================================

This script evaluates trained MaskInterpreter models on cell cycle data and
generates visualizations and predictions showing which image regions are
important for predicting cell cycle markers (Cdt1 and Geminin).

Overview:
---------
The script loads pre-trained MaskInterpreter models and applies them to
cell cycle microscopy images to:
1. Generate importance masks that highlight regions critical for prediction
2. Create visualizations comparing original vs. adapted (masked) images
3. Save mask predictions as .npy files for downstream analysis

This is the evaluation/inference companion to MaskInterpreterRegression.py
which handles training.

Workflow:
---------
    1. Load cell cycle dataset (train/val/test/perturbations)
    2. Normalize images using dataset-specific statistics
    3. Load pretrained regression models (Marker 1 and Marker 2)
    4. Load trained MaskInterpreter models for both markers
    5. Generate importance masks for all images
    6. Optionally create per-sample visualizations
    7. Optionally save all mask predictions as .npy files

Visualization Output:
---------------------
    For each sample, generates a 4-panel figure showing:
    
    Panel 1: Original Image
        - Grayscale brightfield image
        - True marker value and predicted marker value
    
    Panel 2: Adapted (Noisy) Image
        - Image with non-important regions replaced by noise
        - Shows prediction on the adapted image
    
    Panel 3: Importance Mask
        - Heatmap of importance values [0,1]
        - High values (red) = important for prediction
        - Low values (blue) = not important
    
    Panel 4: Mask Overlay
        - Original image with mask overlaid (alpha=0.5)
        - Shows mask efficacy (correlation between original/adapted predictions)

Mask Efficacy:
--------------
    Measured as the Pearson correlation between:
    - Regressor prediction on original image
    - Regressor prediction on adapted image (important regions preserved)
    
    High efficacy (~1.0) = mask correctly identifies important regions
    Low efficacy = mask fails to capture what the regressor uses

Data Format:
------------
    Input Images (.npy):
        - Shape: (T, C, Y, X) - loaded as (64, 64, 1) grayscale
    
    Input Labels (.npy):
        - Shape: (C, T, N) - two markers: Cdt1 (0), Geminin (1)
    
    Output Masks (.npy):
        - Shape: (64, 64) or (64, 64, 1)
        - Values in [0, 1] representing importance

Output Files:
-------------
    Visualizations (if plot_examples=True):
        - ./cellcycle_mi_images_marker1_{dataset}/*.png
        - ./cellcycle_mi_images_marker2_{dataset}/*.png
    
    Predictions (if create_all_predictions=True):
        - {base_dir}/{dataset}/predictions_marker1/*.npy
        - {base_dir}/{dataset}/predictions_marker2/*.npy
        - Filenames match original image filenames

Required Input Models:
----------------------
    Regressors:
        - cellcycle_marker1.h5: Trained Cdt1 regressor
        - cellcycle_marker2.h5: Trained Geminin regressor
    
    MaskInterpreters:
        - ./cellcycle_mi_marker1/: Trained mask interpreter for Marker 1
        - ./cellcycle_mi_marker2/: Trained mask interpreter for Marker 2

Configuration Settings:
-----------------------
    num_samples (int): Number of samples to visualize.
                       Default: 100
    
    dataset_to_visualize (str): Which dataset split to process.
                                Options: "train", "val", "test", "perturbations"
                                Default: "perturbations"
    
    plot_examples (bool): Whether to generate visualization plots.
                          Default: False
    
    create_all_predictions (bool): Whether to save all mask predictions as .npy.
                                   Default: True
    
    adabatch (bool): If True, apply adaptive batch normalization before
                     making predictions on non-training datasets.
                     This adapts the batch normalization statistics to the
                     target dataset (val/test/perturbations) for better
                     prediction accuracy.
                     Default: True

Key Functions:
--------------
    load_single_sample(img_file, labels_dir):
        Load a single image-label pair for parallel processing.
    
    load_cell_cycle_data(data_dir, num_workers=None):
        Load all data from directory using multiprocessing.

Dependencies:
-------------
    - tensorflow / keras
    - numpy
    - matplotlib
    - tqdm
    - multiprocessing
    - Custom modules: metrics (tf_pearson_corr), models.MaskInterpreterRegression, models.UNETO

Usage:
------
    # Edit settings at top of script, then run:
    $ python mi_reg_cellcycle_eval.py
    
    # To only generate predictions (no plots):
    #   plot_examples = False
    #   create_all_predictions = True
    
    # To only visualize samples (no bulk predictions):
    #   plot_examples = True
    #   create_all_predictions = False
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import tf_pearson_corr

###############--SETTINGS--##########################
num_samples = 100  # Number of samples to visualize
dataset_to_visualize = "perturbations"  # Options: "train", "val", "test"
plot_examples = False  # Whether to generate visualization plots
create_all_predictions = True  # Whether to save all mask predictions as .npy files
adabatch = True  # Enable adaptive batch normalization for test/perturbation sets

###############--DATA LOADING--##########################
from models.regressor_cellcycle import get_file_list, compute_dataset_statistics, load_sample_from_file, adapt_batch_norm

if __name__ == "__main__":
    # Load dataset
    base_data_dir = os.path.join(gv.BASE_PATH, 'Gad/Cell_Cycle_Data')
    base_dir = os.path.join(gv.CWD, 'cell_cycle')
    
    print("\n" + "="*60)
    print("Preparing Lazy Loading for Evaluation")
    print("="*60)
    
    # Scan training files to compute normalization statistics
    print("\nScanning training files...")
    train_files, train_samples = get_file_list(os.path.join(base_data_dir, "train"))
    
    # Compute normalization statistics from training set
    print("\nComputing normalization statistics from training data...")
    train_mean, train_std = compute_dataset_statistics(train_files, num_samples=10000)
    
    # Scan the dataset to visualize
    print(f"\nScanning {dataset_to_visualize} files...")
    data_files, data_samples = get_file_list(os.path.join(base_data_dir, dataset_to_visualize))
    
    print(f"\nDataset summary:")
    print(f"  Training samples (for stats): {train_samples}")
    print(f"  {dataset_to_visualize.capitalize()} samples: {data_samples}")
    
    # Load data samples on-demand for evaluation
    print(f"\nLoading {min(num_samples, data_samples)} samples from {dataset_to_visualize} for evaluation...")
    x_data_list = []
    y_data_list = []
    sample_count = 0
    
    for file_info in tqdm(data_files, desc="Loading samples"):
        if sample_count >= num_samples:
            break
        num_timepoints = file_info[2]
        for t in range(num_timepoints):
            if sample_count >= num_samples:
                break
            bf, target = load_sample_from_file(file_info, t)
            # Store original for display
            x_data_list.append(bf)
            y_data_list.append(target)
            sample_count += 1
    
    x_data_orig = np.array(x_data_list)
    y_data = np.array(y_data_list)
    
    # Normalize the data
    print(f"\nNormalizing {len(x_data_orig)} samples...")
    x_data_norm = (x_data_orig - train_mean) / train_std
    
    print(f"\nLoaded data shape: {x_data_orig.shape}")
    
    ###############--LOAD MODELS--##########################
    
    print("\nLoading trained models...")
    
    # Load regressors
    marker1_model = tf.keras.models.load_model(f'{base_dir}/cellcycle_marker1.h5')
    marker2_model = tf.keras.models.load_model(f'{base_dir}/cellcycle_marker2.h5')
    
    # Load mask interpreters
    from models.MaskInterpreterRegression import MaskInterpreterRegression
    from models.UNETO import get_unet
import global_vars as gv
    
    # Create adaptors
    adaptor_m1 = get_unet((64, 64, 2), activation="sigmoid")
    adaptor_m2 = get_unet((64, 64, 2), activation="sigmoid")
    
    # Create mask interpreters
    mi_marker1 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m1,
        regressor=marker1_model,
        pcc_target=0.95
    )
    
    mi_marker2 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m2,
        regressor=marker2_model,
        pcc_target=0.95
    )
    
    # Compile (required before loading weights)
    mi_marker1.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    mi_marker2.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    # Build models
    mi_marker1(np.random.random((1, 64, 64, 1)).astype(np.float32))
    mi_marker2(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    # Load weights using the full model loading approach
    print("Loading Mask Interpreter weights...")
    
    # Load Marker 1 mask interpreter
    mi_marker1_pt = keras.models.load_model(f"{base_dir}/cellcycle_mi_marker1")
    mi_marker1.set_weights(mi_marker1_pt.get_weights())
    print("✓ Loaded Marker 1 mask interpreter weights")
    
    # Load Marker 2 mask interpreter
    mi_marker2_pt = keras.models.load_model(f"{base_dir}/cellcycle_mi_marker2")
    mi_marker2.set_weights(mi_marker2_pt.get_weights())
    print("✓ Loaded Marker 2 mask interpreter weights")

    # Apply adaptive batch normalization if enabled and not training set
    # This is done once here and reused for both visualization and predictions
    if adabatch and dataset_to_visualize != "train":
        print("\n" + "="*60)
        print("Applying Adaptive Batch Normalization...")
        print("="*60)
        print(f"Adapting batch normalization for {dataset_to_visualize} dataset...")
        
        # Use x_data_norm if available (from visualization), otherwise load a batch
        if 'x_data_norm' in locals() and x_data_norm is not None and len(x_data_norm) > 0:
            adapt_batch_norm(marker1_model, x_data_norm, batch_size=32)
            adapt_batch_norm(marker2_model, x_data_norm, batch_size=32)
        else:
            # Load a batch for adaptation (used when only saving predictions)
            print("Loading adaptation batch...")
            adapt_samples = []
            adapt_count = 0
            adapt_batch_size = 1000
            
            for file_info in tqdm(data_files[:min(20, len(data_files))], desc="Loading adaptation batch"):
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
            adapt_batch_norm(marker1_model, adapt_data, batch_size=32)
            adapt_batch_norm(marker2_model, adapt_data, batch_size=32)
            del adapt_data, adapt_samples
        
        print("✓ Batch normalization adaptation complete")

    
    ###############--VISUALIZATION--##########################
    
    if plot_examples:
    
        if plot_examples:
            print("\n" + "="*60)
            print("Generating visualizations...")
            print("="*60)
        
        # Create output directories
        output_dir_m1 = f"{base_dir}/cellcycle_mi_images_marker1_{dataset_to_visualize}"
        output_dir_m2 = f"{base_dir}/cellcycle_mi_images_marker2_{dataset_to_visualize}"
        
        try:
            os.makedirs(output_dir_m1, exist_ok=True)
            os.makedirs(output_dir_m2, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {e}")
        
        sample_indices = range(min(num_samples, x_data.shape[0]))
        
        for idx in tqdm(sample_indices, desc="Visualizing samples"):
            # Get normalized image and its original version for display
            img_norm = x_data_norm[idx]  # normalized image (64, 64, 1)
            img_orig = x_data_orig[idx]  # un-normalized image for plotting
            true_marker1 = y_data[idx, 0]
            true_marker2 = y_data[idx, 1]
            
            ###############--MARKER 1--##########################
            
            # 1. Regressor Prediction on Original Image
            pred_marker1_orig = marker1_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, 0]
            
            # 2. Generate Importance Mask Using the MaskInterpreter
            mask_m1 = mi_marker1(np.expand_dims(img_norm, axis=0))
            mask_m1 = np.squeeze(mask_m1)  # shape becomes (64, 64) or (64, 64, 1)
            
            # Ensure mask is 2D for visualization
            if mask_m1.ndim == 3:
                mask_m1_vis = mask_m1[:, :, 0]
            else:
                mask_m1_vis = mask_m1
                
            # Expand for broadcasting
            if mask_m1.ndim == 2:
                mask_m1_expanded = np.expand_dims(mask_m1, axis=-1)
            else:
                mask_m1_expanded = mask_m1
            
            # 3. Generate a Noisy (Adapted) Image Using the Importance Mask
            noise_m1 = tf.random.normal(shape=img_norm.shape, 
                                     stddev=mi_marker1.noise_scale, 
                                     dtype=tf.float64).numpy()
        
        adapted_image_m1 = (mask_m1_expanded * img_norm.astype(np.float64)) + \
                           ((1 - mask_m1_expanded) * noise_m1)
        
        # 4. Regressor Prediction on Adapted Image
        pred_marker1_adapted = marker1_model.predict(
            np.expand_dims(adapted_image_m1.astype(np.float32), axis=0), verbose=0
        )[0, 0]
        
        # 5. Revert Normalization for Display
        adapted_image_m1_orig = (adapted_image_m1 * std) + mean
        adapted_image_m1_orig = np.clip(adapted_image_m1_orig, 0, 1)
        
        # Calculate mask efficacy (correlation between predictions)
        mask_efficacy_m1 = np.corrcoef([pred_marker1_orig], [pred_marker1_adapted])[0, 1]
        
        # 6. Plot the Results for Marker 1
        plt.figure(figsize=(15, 4))
        
        # Subplot 1: Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.title(f"Original\nTrue Marker1: {true_marker1:.3f}\nPred: {pred_marker1_orig:.3f}")
        plt.axis('off')
        
        # Subplot 2: Adapted (noisy) image
        plt.subplot(1, 4, 2)
        plt.imshow(adapted_image_m1_orig[:, :, 0], cmap='gray')
        plt.title(f"Adapted (Noisy)\nPred: {pred_marker1_adapted:.3f}")
        plt.axis('off')
        
        # Subplot 3: Importance mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_m1_vis, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Importance Mask")
        plt.axis('off')
        
        # Subplot 4: Original image with mask overlay
        plt.subplot(1, 4, 4)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.imshow(mask_m1_vis, cmap='jet', alpha=0.5)
        plt.title(f"Mask Overlay\nEfficacy: {mask_efficacy_m1:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir_m1}/{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        ###############--MARKER 2--##########################
        
        # 1. Regressor Prediction on Original Image
        pred_marker2_orig = marker2_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, 0]
        
        # 2. Generate Importance Mask Using the MaskInterpreter
        mask_m2 = mi_marker2(np.expand_dims(img_norm, axis=0))
        mask_m2 = np.squeeze(mask_m2)
        
        # Ensure mask is 2D for visualization
        if mask_m2.ndim == 3:
            mask_m2_vis = mask_m2[:, :, 0]
        else:
            mask_m2_vis = mask_m2
            
        # Expand for broadcasting
        if mask_m2.ndim == 2:
            mask_m2_expanded = np.expand_dims(mask_m2, axis=-1)
        else:
            mask_m2_expanded = mask_m2
        
        # 3. Generate a Noisy (Adapted) Image
        noise_m2 = tf.random.normal(shape=img_norm.shape, 
                                     stddev=mi_marker2.noise_scale, 
                                     dtype=tf.float64).numpy()
        
        adapted_image_m2 = (mask_m2_expanded * img_norm.astype(np.float64)) + \
                           ((1 - mask_m2_expanded) * noise_m2)
        
        # 4. Regressor Prediction on Adapted Image
        pred_marker2_adapted = marker2_model.predict(
            np.expand_dims(adapted_image_m2.astype(np.float32), axis=0), verbose=0
        )[0, 0]
        
        # 5. Revert Normalization for Display
        adapted_image_m2_orig = (adapted_image_m2 * std) + mean
        adapted_image_m2_orig = np.clip(adapted_image_m2_orig, 0, 1)
        
        # Calculate mask efficacy
        mask_efficacy_m2 = np.corrcoef([pred_marker2_orig], [pred_marker2_adapted])[0, 1]
        
        # 6. Plot the Results for Marker 2
        plt.figure(figsize=(15, 4))
        
        # Subplot 1: Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.title(f"Original\nTrue Marker2: {true_marker2:.3f}\nPred: {pred_marker2_orig:.3f}")
        plt.axis('off')
        
        # Subplot 2: Adapted (noisy) image
        plt.subplot(1, 4, 2)
        plt.imshow(adapted_image_m2_orig[:, :, 0], cmap='gray')
        plt.title(f"Adapted (Noisy)\nPred: {pred_marker2_adapted:.3f}")
        plt.axis('off')
        
        # Subplot 3: Importance mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_m2_vis, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Importance Mask")
        plt.axis('off')
        
        # Subplot 4: Original image with mask overlay
        plt.subplot(1, 4, 4)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.imshow(mask_m2_vis, cmap='jet', alpha=0.5)
        plt.title(f"Mask Overlay\nEfficacy: {mask_efficacy_m2:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir_m2}/{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"\nVisualization complete!")
        print(f"Marker 1 images saved to: {output_dir_m1}")
        print(f"Marker 2 images saved to: {output_dir_m2}")
        print("="*60)
    
    ###############--SAVE ALL PREDICTIONS--##########################
    
    if create_all_predictions:
        print("\n" + "="*60)
        print("Saving all mask predictions...")
        print("="*60)
        
        # Create predictions directories next to images and labels
        base_data_dir = os.path.join(gv.BASE_PATH, 'Gad/Cell_Cycle_Data')
        predictions_dir_m1 = os.path.join(base_data_dir, dataset_to_visualize, "predictions_marker1")
        predictions_dir_m2 = os.path.join(base_data_dir, dataset_to_visualize, "predictions_marker2")
        
        try:
            os.makedirs(predictions_dir_m1, exist_ok=True)
            os.makedirs(predictions_dir_m2, exist_ok=True)
            print(f"Created directories:")
            print(f"  {predictions_dir_m1}")
            print(f"  {predictions_dir_m2}")
        except Exception as e:
            print(f"Error creating prediction directories: {e}")
            exit(1)
        
        # Process all files lazily (batch normalization already adapted above)
        total_predictions = 0
        for file_info in tqdm(data_files, desc="Saving predictions"):
            img_file, label_file, num_timepoints, _ = file_info
            filename = os.path.basename(img_file)
            
            # Load and process each timepoint in this file
            for t in range(num_timepoints):
                bf, _ = load_sample_from_file(file_info, t)
                bf_norm = (bf - train_mean) / train_std
                
                # Generate masks for both markers
                mask_m1 = mi_marker1(np.expand_dims(bf_norm, axis=0))
                mask_m1 = np.squeeze(mask_m1)  # Convert to numpy and remove batch dim
                
                mask_m2 = mi_marker2(np.expand_dims(bf_norm, axis=0))
                mask_m2 = np.squeeze(mask_m2)
                
                # Create timepoint-specific filename
                base_name = os.path.splitext(filename)[0]
                pred_filename = f"{base_name}_t{t:04d}.npy"
                
                # Save masks
                np.save(os.path.join(predictions_dir_m1, pred_filename), mask_m1)
                np.save(os.path.join(predictions_dir_m2, pred_filename), mask_m2)
                total_predictions += 1
        
        print(f"\nPrediction saving complete!")
        print(f"Marker 1 masks saved to: {predictions_dir_m1}")
        print(f"Marker 2 masks saved to: {predictions_dir_m2}")
        print(f"Total predictions saved: {total_predictions} per marker")
        print("="*60)

