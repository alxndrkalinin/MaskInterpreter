"""
Cell Cycle Regression Model
===========================

This script trains ResNet18-based regression models to predict cell cycle markers
(Cdt1 and Geminin) from brightfield microscopy images of cells.

Overview:
---------
The script performs the following steps:
    1. Load cell cycle dataset (train/val/test/perturbations splits)
    2. Normalize images using per-dataset statistics
    3. Build and train two separate ResNet18-based regression models:
       - Marker 1 model: Predicts Cdt1 intensity (cell cycle G1/S marker)
       - Marker 2 model: Predicts Geminin intensity (cell cycle S/G2/M marker)
    4. Evaluate models on test and perturbation datasets
    5. Generate visualization plots (scatter, residual, distribution plots)

Data Format:
------------
    Images (.npy files):
        - Shape: (T, C, Y, X)
        - T = number of timepoints
        - C = channel dimension (only first channel used for brightfield)
        - Y, X = spatial dimensions (e.g., 64x64 pixels)
    
    Labels (.npy files):
        - Shape: (2, T, C)
        - 2 = number of markers (Cdt1, Geminin)
        - T = number of timepoints
        - C = number of channels/cells (only first cell used)
    
    After loading:
        - Each sample = one timepoint
        - Images: (N, Y, X, 1) - single-channel brightfield
        - Labels: (N, 2) - two marker values (Cdt1, Geminin)

Data Loading:
-------------
    The load_cell_cycle_data() function loads ALL timepoints:
    
    >>> x_train, y_train = load_cell_cycle_data(data_dir)
    
    Each timepoint becomes a separate sample.
    - Images shape: (N, Y, X, 1) - single-channel brightfield
    - Labels shape: (N, 2) where 2 = the two markers (Cdt1, Geminin)
    
    The model processes a single brightfield image and predicts both marker values.

Model Architecture:
-------------------
    ResNet18-based architecture with custom modifications:
    
    1. Custom Stem (for grayscale input):
       - Conv2D(64, 7x7, stride=2) -> BatchNorm -> ReLU -> MaxPool
    
    2. ResNet18 Backbone:
       - Block 1: 2x BasicBlock(64 filters)
       - Block 2: 2x BasicBlock(128 filters, first with stride=2)
       - Block 3: 2x BasicBlock(256 filters, first with stride=2)
       - Block 4: 2x BasicBlock(512 filters, first with stride=2)
       - Global Average Pooling -> 512-dim feature vector
    
    3. MLP Head:
       - Dense(512) embedding layer
       - 4x [Dense(512) -> LayerNorm -> GELU] blocks
       - Dense(1, sigmoid) output layer
    
    Output: Single value in [0, 1] representing marker intensity

Adaptive Batch Normalization:
-----------------------------
    When adabatch=True (default), batch normalization statistics are
    adapted to each evaluation dataset before predictions. This helps
    account for domain shift between training and test/perturbation data.
    
    The adapt_batch_norm() function runs forward passes with training=True
    to update running mean/variance statistics in BN layers.

Training Configuration:
-----------------------
    - Optimizer: Adam (learning_rate=1e-4)
    - Loss: Mean Squared Error (MSE)
    - Metrics: Mean Absolute Error (MAE)
    - Batch size: 32
    - Max epochs: 100
    - Early stopping: patience=10, restore_best_weights=True

Output Files:
-------------
    Models:
        - cellcycle_marker1.h5: Trained Cdt1 prediction model
        - cellcycle_marker2.h5: Trained Geminin prediction model
    
    Visualizations:
        - cellcycle_markers_results.png: Scatter and residual plots (test set)
        - cellcycle_cdt1_distribution.png: Cdt1 prediction distribution
        - cellcycle_geminin_distribution.png: Geminin prediction distribution
        - cellcycle_joint_distribution.png: Joint Cdt1-Geminin density (predicted)
        - cellcycle_joint_distribution_true.png: Joint density (ground truth)
        - cellcycle_markers_perturbations_results.png: Perturbation results
        - cellcycle_test_vs_perturbations_comparison.png: MAE comparison bar chart

Configuration Flags:
--------------------
    load (bool): If True, load pre-trained models instead of training.
                 Default: False (train new models)
    
    adabatch (bool): If True, apply adaptive batch normalization before
                     evaluation on each dataset. Default: True

Usage Example:
--------------
    # Train new models
    $ python cellcycle-clf.py
    
    # To load existing models instead of training, set load=True in the script

Dependencies:
-------------
    - tensorflow / keras
    - numpy
    - matplotlib
    - multiprocessing
    - tqdm
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import gc
from sklearn.preprocessing import StandardScaler
import global_vars as gv

# Configuration variables
load = False  # If True, load pre-trained models instead of training
adabatch = True  # Enable adaptive batch normalization for test/perturbation sets
batch_size = 32  # Batch size for training and evaluation
base_dir = os.path.join(gv.CWD, 'cell_cycle')  # Directory for saving models and visualizations
shuffle_seed = 42  # Random seed for reproducibility


def load_single_sample(img_file, labels_dir, bf_channel=0):
        """Load image-label pairs from a file. Used for parallel processing.
        
        Loads ALL timepoints from each file. Each timepoint becomes a separate sample.
        Images are single-channel brightfield, labels contain the two marker values.
        
        Args:
            img_file: Path to the image file
            labels_dir: Directory containing label files
            bf_channel: Which channel to use for brightfield (default: 0)
        
        Returns:
            Tuple of (list of images, list of labels) where each image has shape (Y, X)
            and each label has shape (2,) for the two markers (Cdt1, Geminin)
        """
        try:
            filename = os.path.basename(img_file)
            label_file = os.path.join(labels_dir, filename)
            
            if not os.path.exists(label_file):
                return None, None
            
            # Load full arrays
            img_data = np.load(img_file)  # Shape: (T, C, Y, X)
            label_data = np.load(label_file)  # Shape: (2, T, N) where 2=markers (Cdt1, Geminin), T=timepoints, N=cell indices
            # print(f"Loaded {filename}: img shape {img_data.shape}, label shape {label_data.shape}")
            # Verify dimensions match
            num_timepoints = img_data.shape[0]  # T dimension
            # num_channels = img_data.shape[1]  # C dimension (cell indices/channels)
            
            # Label dimensions
            num_markers = label_data.shape[0]  # Should be 2 (Cdt1, Geminin)
            label_timepoints = label_data.shape[1]  # Should match num_timepoints
            label_channels = label_data.shape[2]  # Should match num_channels
            
            # Sanity check
            if label_timepoints != num_timepoints:
                print(f"Warning: Timepoint mismatch in {filename}: img={num_timepoints}, label={label_timepoints}")
                return None, None
            # if label_channels != num_channels:
            #     print(f"Warning: Channel mismatch in {filename}: img={num_channels}, label={label_channels}")
            #     return None, None
            
            images = []
            labels = []
            
            # Load all timepoints as separate samples
            # Images are single-channel (Y, X), labels are (2,) for the two markers
            for t in range(num_timepoints):
                # Image: (T, C, Y, X) -> take timepoint t, bf_channel (brightfield is single channel)
                bf = img_data[t, bf_channel]  # Shape: (Y, X) - single brightfield image at timepoint t
                # Labels: (2, T, N) -> take all markers, timepoint t, first cell/channel
                target = label_data[:, t, 0]  # Shape: (2,) - both markers (Cdt1, Geminin) for first cell
                images.append(bf)
                labels.append(target)
            
            return images, labels
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            return None, None

def load_cell_cycle_data(data_dir, num_workers=None, bf_channel=0):
    """Load all cell cycle data from the given directory using parallel processing.
    
    Loads ALL timepoints from all files. Each timepoint becomes a separate sample.
    Images are single-channel brightfield, labels contain the two marker values.
    
    Args:
        data_dir: Directory containing images/ and labels/ subdirectories
        num_workers: Number of parallel workers (default: cpu_count - 1)
        bf_channel: Which channel to use for brightfield (default: 0)
    
    Returns:
        Tuple of (images, labels) as numpy arrays where:
        - images shape: (N, Y, X, 1) - N samples, single-channel grayscale
        - labels shape: (N, 2) - N samples, 2 markers (Cdt1, Geminin)
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
    
    print(f"  Found {len(image_files)} files in {data_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"  Loading with {num_workers} workers...")
    
    # Parallel loading with progress bar
    load_func = partial(load_single_sample, labels_dir=labels_dir, bf_channel=bf_channel)
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_func, image_files),
            total=len(image_files),
            desc="  Loading files",
            unit="file"
        ))
    
    # Filter out failed loads and separate images and labels
    all_images = []
    all_labels = []
    
    for bf, target in results:
        if bf is not None and target is not None:
            # bf and target are lists containing all timepoints and indices
            all_images.extend(bf)
            all_labels.extend(target)
    
    # Convert to numpy arrays
    images = np.array(all_images)  # Shape: (N, Y, X)
    labels = np.array(all_labels)  # Shape: (N, 2)
    
    # Add channel dimension for single-channel grayscale images
    images = np.expand_dims(images, axis=-1)  # (N, Y, X) -> (N, Y, X, 1)
    
    print(f"  Loaded {len(images)} samples")
    print(f"  Image shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return images, labels

def fit_scaler(images, batch_size_norm=1000):
    """
    Fit a StandardScaler on image data using incremental learning.
    
    Args:
        images: numpy array of shape (N, H, W, C) - images to fit scaler on
        batch_size_norm: batch size for incremental fitting
    
    Returns:
        Fitted StandardScaler instance
    """
    print(f"Fitting scaler on {len(images)} samples...")
    scaler = StandardScaler()
    
    # Process in-place without copying entire array
    original_shape = images.shape
    n_samples = original_shape[0]
    n_features = np.prod(original_shape[1:])
    
    # Fit in batches to save memory
    for i in range(0, n_samples, batch_size_norm):
        end_idx = min(i + batch_size_norm, n_samples)
        batch = images[i:end_idx].reshape(end_idx - i, -1).astype("float32")
        scaler.partial_fit(batch)
        del batch
        if i % (batch_size_norm * 10) == 0:
            print(f"  Processed {i}/{n_samples} samples for fitting")
    
    print(f"Scaler fitted - Mean: {scaler.mean_[0]:.4f}, Std: {np.sqrt(scaler.var_[0]):.4f}")
    return scaler

def transform_with_scaler(images, scaler, batch_size_norm=1000, verbose=True):
    """
    Transform image data using a fitted StandardScaler in-place.
    
    Args:
        images: numpy array of shape (N, H, W, C) - images to transform (modified in-place)
        scaler: Fitted StandardScaler instance
        batch_size_norm: batch size for batch-wise transformation
        verbose: whether to print progress messages
    
    Returns:
        Transformed images as numpy array (same object, modified in-place)
    """
    if verbose:
        print(f"Transforming {len(images)} samples with scaler (in-place)...")
    
    original_shape = images.shape
    n_samples = original_shape[0]
    
    # Transform in batches, modifying in-place
    for i in range(0, n_samples, batch_size_norm):
        end_idx = min(i + batch_size_norm, n_samples)
        batch = images[i:end_idx].reshape(end_idx - i, -1).astype("float32")
        batch_transformed = scaler.transform(batch)
        images[i:end_idx] = batch_transformed.reshape(images[i:end_idx].shape)
        del batch, batch_transformed
        if verbose and i % (batch_size_norm * 10) == 0:
            print(f"  Transformed {i}/{n_samples} samples")
    
    gc.collect()
    return images

# ===============================================================
# LAZY LOADING DATASET IMPLEMENTATION
# ===============================================================

def get_file_list(data_dir, bf_channel=0):
    """Get list of image-label file pairs with sample counts."""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
    file_pairs = []
    total_samples = 0
    
    print(f"  Scanning files in {data_dir}...")
    for img_file in tqdm(image_files, desc="  Scanning", unit="file"):
        filename = os.path.basename(img_file)
        label_file = os.path.join(labels_dir, filename)
        
        if not os.path.exists(label_file):
            continue
        
        try:
            # Just load shape, not data
            img_shape = np.load(img_file, mmap_mode='r').shape
            num_timepoints = img_shape[0]
            file_pairs.append((img_file, label_file, num_timepoints, bf_channel))
            total_samples += num_timepoints
        except Exception as e:
            print(f"  Warning: Error scanning {img_file}: {e}")
            continue
    
    print(f"  Found {len(file_pairs)} valid files with {total_samples} total samples")
    return file_pairs, total_samples

def load_sample_from_file(file_info, timepoint_idx):
    """Load a single sample (one timepoint) from a file.
    
    Args:
        file_info: Tuple of (img_file, label_file, num_timepoints, bf_channel)
        timepoint_idx: Which timepoint to load from the file
    
    Returns:
        Tuple of (image, label) where image is (64, 64, 1) and label is (2,)
    """
    img_file, label_file, _, bf_channel = file_info
    
    # Load only the specific timepoint needed
    img_data = np.load(img_file, mmap_mode='r')
    label_data = np.load(label_file, mmap_mode='r')
    
    # Extract single timepoint
    bf = img_data[timepoint_idx, bf_channel]  # (Y, X)
    target = label_data[:, timepoint_idx, 0]  # (2,)
    
    # Add channel dimension
    bf = np.expand_dims(bf, axis=-1)  # (Y, X, 1)
    
    return bf.astype(np.float32), target.astype(np.float32)

def compute_dataset_statistics(file_pairs, num_samples=10000):
    """Compute mean and std from a subset of the data for normalization."""
    print(f"  Computing normalization statistics from {num_samples} samples...")
    
    # Sample uniformly across files
    samples = []
    samples_per_file = max(1, num_samples // len(file_pairs))
    
    for file_info in tqdm(file_pairs[:min(len(file_pairs), num_samples // samples_per_file)], 
                        desc="  Sampling", unit="file"):
        img_file, _, num_timepoints, bf_channel = file_info
        
        # Sample a few timepoints from this file
        timepoint_indices = np.linspace(0, num_timepoints-1, 
                                    min(samples_per_file, num_timepoints), 
                                    dtype=int)
        
        for t in timepoint_indices:
            bf, _ = load_sample_from_file(file_info, t)
            samples.append(bf.flatten())
            
            if len(samples) >= num_samples:
                break
        
        if len(samples) >= num_samples:
            break
    
    # Compute statistics
    all_samples = np.concatenate(samples)
    mean = np.mean(all_samples)
    std = np.std(all_samples)
    
    print(f"  Computed statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std

def create_lazy_dataset(file_pairs, mean, std, shuffle=True, batch_size=64, 
                    prefetch_buffer=tf.data.AUTOTUNE, cache=False):
    """Create a lazy-loading tf.data.Dataset.
    
    Args:
        file_pairs: List of (img_file, label_file, num_timepoints, bf_channel) tuples
        mean: Mean for normalization
        std: Std for normalization
        shuffle: Whether to shuffle the dataset
        batch_size: Batch size for training
        prefetch_buffer: Prefetch buffer size
        cache: Whether to cache the dataset in memory (only for small datasets)
    
    Returns:
        tf.data.Dataset that yields batches of (images, labels)
    """
    # Create a flat list of all (file_info, timepoint_idx) pairs
    all_samples = []
    for file_info in file_pairs:
        num_timepoints = file_info[2]
        for t in range(num_timepoints):
            all_samples.append((file_info, t))
    
    total_samples = len(all_samples)
    print(f"  Creating dataset with {total_samples} samples, batch_size={batch_size}")
    
    # Create dataset from generator
    def data_generator():
        indices = np.arange(len(all_samples))
        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(indices)
        for idx in indices:
            file_info, t = all_samples[idx]
            bf, target = load_sample_from_file(file_info, t)
            # Normalize
            bf = (bf - mean) / std
            yield bf, target
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    
    if cache:
        dataset = dataset.cache()
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, total_samples), seed=shuffle_seed)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely for multiple epochs
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset, total_samples

def adapt_batch_norm(model, data, batch_size=32):
    """
    Adapt batch normalization statistics to new data.
    Sets BN layers to training mode to update running statistics.
    
    Args:
        model: Keras model with BatchNormalization layers
        data: Data to adapt to (numpy array)
        batch_size: Batch size for adaptation
    """
    print(f"  Adapting batch normalization statistics on {len(data)} samples...")
    
    # Find all BatchNormalization layers
    bn_layers = [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]
    
    if not bn_layers:
        print("  Warning: No BatchNormalization layers found in model.")
        return
    
    # Create a temporary model that runs in training mode for BN layers only
    # We'll do a forward pass with training=True to update BN statistics
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]
        
        # Forward pass with training=True to update BN running statistics
        _ = model(batch, training=True)
    
    print(f"  Adapted {len(bn_layers)} BatchNormalization layers")

def create_regression_model(input_shape, name="marker_model"):
    """Create a ResNet18-based regression model for cell cycle marker prediction.
    
    Architecture inspired by the original PyTorch model:
    - ResNet18 backbone (modified for grayscale input)
    - MLP head with LayerNorm and GELU activations
    - Outputs single value for regression
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Option 1: Replicate single channel to 3 channels (like original model)
    # x = layers.Concatenate()([inputs, inputs, inputs])  # (64, 64, 3)
    
    # Option 2: Use single channel directly with modified ResNet18
    # We'll use single channel since it's more efficient
    
    # Load ResNet18 base (without top, without pretrained weights)
    # Note: TensorFlow's ResNet expects 3 channels, so we'll build custom stem
    
    # Custom stem for grayscale input (replacing ResNet's first conv)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', 
                    use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # ResNet18 blocks (manually constructed)
    # Block 1: 64 channels, no downsampling
    x = _resnet_block(x, 64, stride=1, name='block1a')
    x = _resnet_block(x, 64, stride=1, name='block1b')
    
    # Block 2: 128 channels, downsample
    x = _resnet_block(x, 128, stride=2, name='block2a')
    x = _resnet_block(x, 128, stride=1, name='block2b')
    
    # Block 3: 256 channels, downsample
    x = _resnet_block(x, 256, stride=2, name='block3a')
    x = _resnet_block(x, 256, stride=1, name='block3b')
    
    # Block 4: 512 channels, downsample
    x = _resnet_block(x, 512, stride=2, name='block4a')
    x = _resnet_block(x, 512, stride=1, name='block4b')
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)  # -> (batch, 512)
    
    # MLP Head (like the original PyTorch model)
    # Embedding layer
    x = layers.Dense(512, name='head_embed')(x)
    
    # 4 MLP blocks with LayerNorm and GELU
    for i in range(4):
        x = layers.Dense(512, name=f'head_block{i}_dense')(x)
        x = layers.LayerNormalization(name=f'head_block{i}_ln')(x)
        x = layers.Activation('gelu', name=f'head_block{i}_gelu')(x)
    
    # Final projection to output
    outputs = layers.Dense(1, activation='sigmoid', name='head_proj')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model

def _resnet_block(x, filters, stride=1, name='block'):
    """Create a ResNet basic block (used in ResNet18/34)."""
    shortcut = x
    
    # First conv
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                    use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    
    # Second conv
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                    use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Shortcut connection (with projection if needed)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                padding='same', use_bias=False,
                                name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    
    # Add shortcut and apply ReLU
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)
    
    return x

def marker1_dataset(file_pairs, mean, std, shuffle, batch_size, seed=None):
    """Create dataset that yields only marker 1 (first marker)."""
    def data_generator():
        all_samples = []
        for file_info in file_pairs:
            num_timepoints = file_info[2]
            for t in range(num_timepoints):
                all_samples.append((file_info, t))
        indices = np.arange(len(all_samples))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for idx in indices:
            file_info, t = all_samples[idx]
            bf, target = load_sample_from_file(file_info, t)
            bf = (bf - mean) / std
            yield bf, target[0]  # Only marker 1
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely for multiple epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def marker2_dataset(file_pairs, mean, std, shuffle, batch_size, seed=None):
    """Create dataset that yields only marker 2 (second marker)."""
    def data_generator():
        all_samples = []
        for file_info in file_pairs:
            num_timepoints = file_info[2]
            for t in range(num_timepoints):
                all_samples.append((file_info, t))
        indices = np.arange(len(all_samples))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for idx in indices:
            file_info, t = all_samples[idx]
            bf, target = load_sample_from_file(file_info, t)
            bf = (bf - mean) / std
            yield bf, target[1]  # Only marker 2
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Repeat indefinitely for multiple epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_eval_dataset(file_pairs, mean, std, marker_idx, bs=None):
    """Create dataset for evaluation that yields single marker."""
    if bs is None:
        bs = batch_size
    def data_generator():
        for file_info in file_pairs:
            num_timepoints = file_info[2]
            for t in range(num_timepoints):
                bf, target = load_sample_from_file(file_info, t)
                bf = (bf - mean) / std
                yield bf, target[marker_idx]
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_pred_distributions_01(Y_pred, Y_true=None, bins=60, title_prefix=""):
    """
    Plot distribution histograms and joint density for cell cycle markers.
    
    Y_pred: (N,2) in [0,1]   -> [Cdt1, Geminin]
    Y_true: optional (N,2)   -> overlay as outline
    """
    # clamp just in case
    P = np.clip(Y_pred.astype(float), 0.0, 1.0)
    T = np.clip(Y_true.astype(float), 0.0, 1.0) if Y_true is not None else None

    # 1) Cdt1 (Marker 1) histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(P[:,0], bins=bins, range=(0,1), density=True, alpha=0.7, label="pred", color='skyblue')
    if T is not None:
        ax.hist(T[:,0], bins=bins, range=(0,1), density=True, histtype="step", linewidth=1.5, label="true", color='darkblue')
    ax.set(xlabel="Cdt1 intensity (0–1)", ylabel="Density", title=f"{title_prefix}Cdt1 Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_cdt1_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2) Geminin (Marker 2) histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(P[:,1], bins=bins, range=(0,1), density=True, alpha=0.7, label="pred", color='lightcoral')
    if T is not None:
        ax.hist(T[:,1], bins=bins, range=(0,1), density=True, histtype="step", linewidth=1.5, label="true", color='darkred')
    ax.set(xlabel="Geminin intensity (0–1)", ylabel="Density", title=f"{title_prefix}Geminin Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_geminin_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3) Joint distribution (2D hexbin)
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(P[:,0], P[:,1], gridsize=50, extent=(0,1,0,1), bins="log", cmap='viridis')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log(count)")
    ax.set(xlabel="Cdt1 (0–1)", ylabel="Geminin (0–1)", title=f"{title_prefix}Joint Density (Predicted)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_joint_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4) Joint distribution for true values (if available)
    if T is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        hb = ax.hexbin(T[:,0], T[:,1], gridsize=50, extent=(0,1,0,1), bins="log", cmap='plasma')
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log(count)")
        ax.set(xlabel="Cdt1 (0–1)", ylabel="Geminin (0–1)", title=f"{title_prefix}Joint Density (True)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_dir}/cellcycle_joint_distribution_true.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Load pre-split datasets from train/val/test directories
    base_data_dir = os.path.join(gv.BASE_PATH, 'Gad/Cell_Cycle_Data')

    print("\n" + "="*60)
    print("Preparing Training Set (Lazy Loading)")
    print("="*60)
    train_files, train_samples = get_file_list(os.path.join(base_data_dir, "train"))

    # Compute normalization statistics from training set
    train_mean, train_std = compute_dataset_statistics(train_files, num_samples=10000)

    print("\n" + "="*60)
    print("Preparing Validation Set (Lazy Loading)")
    print("="*60)
    val_files, val_samples = get_file_list(os.path.join(base_data_dir, "val"))

    print("\n" + "="*60)
    print("Preparing Test Set (Lazy Loading)")
    print("="*60)
    test_files, test_samples = get_file_list(os.path.join(base_data_dir, "test"))

    print("\n" + "="*60)
    print("Preparing Perturbations Set (Lazy Loading)")
    print("="*60)
    pert_files, pert_samples = get_file_list(os.path.join(base_data_dir, "perturbations"))

    print(f"\nDataset summary:")
    print(f"  Training samples: {train_samples}")
    print(f"  Validation samples: {val_samples}")
    print(f"  Test samples: {test_samples}")
    print(f"  Perturbations samples: {pert_samples}")

    # Set input shape
    input_shape = (64, 64, 1)
    print(f"Input shape: {input_shape}")

    # 2. Build regression model for Marker 1
    print("\n" + "="*60)
    print("Building Marker 1 Regression Model...")
    print("="*60)

    # Create and compile Marker 1 model
    marker1_model = create_regression_model(input_shape, name="marker1_model")
    marker1_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    marker1_model.summary()

    # Early stopping callback
    early_stopping_m1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    save_path_m1 = f"{base_dir}/cellcycle_marker1.h5"

    if not load:
        print("\nCreating training and validation datasets for Marker 1...")
        
        train_ds_m1 = marker1_dataset(train_files, train_mean, train_std, True, batch_size, seed=shuffle_seed)
        val_ds_m1 = marker1_dataset(val_files, train_mean, train_std, False, batch_size, seed=None)
        
        print("\nTraining Marker 1 model...")
        history_m1 = marker1_model.fit(
            train_ds_m1,
            epochs=100,
            steps_per_epoch=train_samples // batch_size,
            validation_data=val_ds_m1,
            validation_steps=val_samples // batch_size,
            callbacks=[early_stopping_m1],
            verbose=1
        )
        marker1_model.save(save_path_m1)
        gc.collect()
    else:
        marker1_model = tf.keras.models.load_model(save_path_m1)

    print(f"Marker 1 model saved to {save_path_m1}")

    # 3. Build regression model for Marker 2
    print("\n" + "="*60)
    print("Building Marker 2 Regression Model...")
    print("="*60)

    marker2_model = create_regression_model(input_shape, name="marker2_model")
    marker2_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )

    # Early stopping callback
    early_stopping_m2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    save_path_m2 = f"{base_dir}/cellcycle_marker2.h5"

    if not load:
        print("\nCreating training and validation datasets for Marker 2...")
        
        train_ds_m2 = marker2_dataset(train_files, train_mean, train_std, True, batch_size, seed=shuffle_seed)
        val_ds_m2 = marker2_dataset(val_files, train_mean, train_std, False, batch_size, seed=None)
        
        print("\nTraining Marker 2 model...")
        history_m2 = marker2_model.fit(
            train_ds_m2,
            epochs=100,
            steps_per_epoch=train_samples // batch_size,
            validation_data=val_ds_m2,
            validation_steps=val_samples // batch_size,
            callbacks=[early_stopping_m2],
            verbose=1
        )
        marker2_model.save(save_path_m2)
        gc.collect()
    else:
        marker2_model = tf.keras.models.load_model(save_path_m2)

    print(f"Marker 2 model saved to {save_path_m2}")

    # 4. Evaluate both models
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\nCreating evaluation datasets...")

    # Marker 1 Evaluation
    print("\nMarker 1 Results:")
    print("-" * 40)

    train_ds_m1_eval = create_eval_dataset(train_files, train_mean, train_std, 0)
    m1_train_loss, m1_train_mae = marker1_model.evaluate(train_ds_m1_eval, verbose=0, steps=train_samples // batch_size)
    print(f"Training   - MSE: {m1_train_loss:.6f}, MAE: {m1_train_mae:.6f}")

    val_ds_m1_eval = create_eval_dataset(val_files, train_mean, train_std, 0)
    m1_val_loss, m1_val_mae = marker1_model.evaluate(val_ds_m1_eval, verbose=0, steps=val_samples // batch_size)
    print(f"Validation - MSE: {m1_val_loss:.6f}, MAE: {m1_val_mae:.6f}")

    test_ds_m1_eval = create_eval_dataset(test_files, train_mean, train_std, 0)
    m1_test_loss, m1_test_mae = marker1_model.evaluate(test_ds_m1_eval, verbose=0, steps=test_samples // batch_size)
    print(f"Test       - MSE: {m1_test_loss:.6f}, MAE: {m1_test_mae:.6f}")

    # Marker 2 Evaluation
    print("\nMarker 2 Results:")
    print("-" * 40)

    train_ds_m2_eval = create_eval_dataset(train_files, train_mean, train_std, 1)
    m2_train_loss, m2_train_mae = marker2_model.evaluate(train_ds_m2_eval, verbose=0, steps=train_samples // batch_size)
    print(f"Training   - MSE: {m2_train_loss:.6f}, MAE: {m2_train_mae:.6f}")

    val_ds_m2_eval = create_eval_dataset(val_files, train_mean, train_std, 1)
    m2_val_loss, m2_val_mae = marker2_model.evaluate(val_ds_m2_eval, verbose=0, steps=val_samples // batch_size)
    print(f"Validation - MSE: {m2_val_loss:.6f}, MAE: {m2_val_mae:.6f}")

    test_ds_m2_eval = create_eval_dataset(test_files, train_mean, train_std, 1)
    m2_test_loss, m2_test_mae = marker2_model.evaluate(test_ds_m2_eval, verbose=0, steps=test_samples // batch_size)
    print(f"Test       - MSE: {m2_test_loss:.6f}, MAE: {m2_test_mae:.6f}")

    # Perturbations Evaluation
    print("\n" + "="*60)
    print("PERTURBATIONS RESULTS")
    print("="*60)

    # Marker 1 on Perturbations
    print("\nMarker 1 Results (Perturbations):")
    print("-" * 40)
    pert_ds_m1_eval = create_eval_dataset(pert_files, train_mean, train_std, 0)
    m1_pert_loss, m1_pert_mae = marker1_model.evaluate(pert_ds_m1_eval, verbose=0, steps=pert_samples // batch_size)
    print(f"Perturbations - MSE: {m1_pert_loss:.6f}, MAE: {m1_pert_mae:.6f}")

    # Marker 2 on Perturbations
    print("\nMarker 2 Results (Perturbations):")
    print("-" * 40)
    pert_ds_m2_eval = create_eval_dataset(pert_files, train_mean, train_std, 1)
    m2_pert_loss, m2_pert_mae = marker2_model.evaluate(pert_ds_m2_eval, verbose=0, steps=pert_samples // batch_size)
    print(f"Perturbations - MSE: {m2_pert_loss:.6f}, MAE: {m2_pert_mae:.6f}")

    # Compare test vs perturbations performance
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: Test vs Perturbations")
    print("="*60)
    print(f"\nMarker 1 (Cdt1):")
    print(f"  Test MAE:          {m1_test_mae:.6f}")
    print(f"  Perturbations MAE: {m1_pert_mae:.6f}")
    print(f"  Difference:        {m1_pert_mae - m1_test_mae:.6f} ({((m1_pert_mae/m1_test_mae - 1)*100):.2f}%)")

    print(f"\nMarker 2 (Geminin):")
    print(f"  Test MAE:          {m2_test_mae:.6f}")
    print(f"  Perturbations MAE: {m2_pert_mae:.6f}")
    print(f"  Difference:        {m2_pert_mae - m2_test_mae:.6f} ({((m2_pert_mae/m2_test_mae - 1)*100):.2f}%)")

    # 5. Visualize predictions on test set
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # Load a subset of test data for visualization (to avoid OOM)
    print("Loading subset of test data for visualization (10000 samples)...")
    viz_samples = []
    viz_labels = []
    viz_count = 0
    max_viz_samples = 10000

    for file_info in test_files:
        if viz_count >= max_viz_samples:
            break
        num_timepoints = file_info[2]
        for t in range(num_timepoints):
            if viz_count >= max_viz_samples:
                break
            bf, target = load_sample_from_file(file_info, t)
            bf = (bf - train_mean) / train_std
            viz_samples.append(bf)
            viz_labels.append(target)
            viz_count += 1

    x_test_viz = np.array(viz_samples)
    y_test_viz = np.array(viz_labels)
    print(f"Loaded {len(x_test_viz)} samples for visualization")

    # Get predictions
    print("Generating predictions...")
    m1_test_pred = marker1_model.predict(x_test_viz, batch_size=batch_size, verbose=0).flatten()
    m2_test_pred = marker2_model.predict(x_test_viz, batch_size=batch_size, verbose=0).flatten()

    # Combine predictions into (N, 2) array for distribution plots
    Y_pred = np.stack([m1_test_pred, m2_test_pred], axis=1)
    Y_true = y_test_viz

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Marker 1 - Scatter plot
    axes[0, 0].scatter(y_test_viz[:, 0], m1_test_pred, alpha=0.5)
    axes[0, 0].plot([y_test_viz[:, 0].min(), y_test_viz[:, 0].max()], 
                    [y_test_viz[:, 0].min(), y_test_viz[:, 0].max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Marker 1 Values')
    axes[0, 0].set_ylabel('Predicted Marker 1 Values')
    axes[0, 0].set_title(f'Marker 1 Predictions (Test MAE: {m1_test_mae:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Marker 1 - Residuals
    residuals_m1 = y_test_viz[:, 0] - m1_test_pred
    axes[0, 1].scatter(m1_test_pred, residuals_m1, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Marker 1 Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Marker 1 Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Marker 2 - Scatter plot
    axes[1, 0].scatter(y_test_viz[:, 1], m2_test_pred, alpha=0.5)
    axes[1, 0].plot([y_test_viz[:, 1].min(), y_test_viz[:, 1].max()], 
                    [y_test_viz[:, 1].min(), y_test_viz[:, 1].max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('True Marker 2 Values')
    axes[1, 0].set_ylabel('Predicted Marker 2 Values')
    axes[1, 0].set_title(f'Marker 2 Predictions (Test MAE: {m2_test_mae:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Marker 2 - Residuals
    residuals_m2 = y_test_viz[:, 1] - m2_test_pred
    axes[1, 1].scatter(m2_test_pred, residuals_m2, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Marker 2 Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Marker 2 Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_markers_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to '{base_dir}/cellcycle_markers_results.png'")
    plt.show()

    # 6. Distribution plots for test set predictions
    print("\n" + "="*60)
    print("Generating distribution plots...")
    print("="*60)

    # Generate distribution plots
    plot_pred_distributions_01(Y_pred, Y_true=Y_true, bins=60, title_prefix="Test Set - ")
    print("\nDistribution plots saved:")
    print(f"  - {base_dir}/cellcycle_cdt1_distribution.png")
    print(f"  - {base_dir}/cellcycle_geminin_distribution.png")
    print(f"  - {base_dir}/cellcycle_joint_distribution.png")
    print(f"  - {base_dir}/cellcycle_joint_distribution_true.png")

    # 7. Visualize predictions on perturbations set
    print("\n" + "="*60)
    print("Generating perturbations visualizations...")
    print("="*60)

    # Load a subset of perturbations data for visualization (to avoid OOM)
    print("Loading subset of perturbations data for visualization (10000 samples)...")
    pert_viz_samples = []
    pert_viz_labels = []
    pert_viz_count = 0
    max_pert_viz_samples = 10000

    for file_info in pert_files:
        if pert_viz_count >= max_pert_viz_samples:
            break
        num_timepoints = file_info[2]
        for t in range(num_timepoints):
            if pert_viz_count >= max_pert_viz_samples:
                break
            bf, target = load_sample_from_file(file_info, t)
            bf = (bf - train_mean) / train_std
            pert_viz_samples.append(bf)
            pert_viz_labels.append(target)
            pert_viz_count += 1

    x_pert = np.array(pert_viz_samples)
    y_pert = np.array(pert_viz_labels)
    print(f"Loaded {len(x_pert)} perturbation samples for visualization")

    # Re-adapt for perturbations if adabatch is enabled
    if adabatch:
        print("\nRe-adapting batch normalization for perturbations predictions...")
        adapt_batch_norm(marker1_model, x_pert, batch_size=32)
        adapt_batch_norm(marker2_model, x_pert, batch_size=32)

    # Get predictions for perturbations
    print("Generating perturbation predictions...")
    m1_pert_pred = marker1_model.predict(x_pert, batch_size=batch_size, verbose=0).flatten()
    m2_pert_pred = marker2_model.predict(x_pert, batch_size=batch_size, verbose=0).flatten()

    # Combine predictions
    Y_pred_pert = np.stack([m1_pert_pred, m2_pert_pred], axis=1)
    Y_true_pert = y_pert

    # Create scatter and residual plots for perturbations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Marker 1 - Scatter plot
    axes[0, 0].scatter(y_pert[:, 0], m1_pert_pred, alpha=0.5, color='orange')
    axes[0, 0].plot([y_pert[:, 0].min(), y_pert[:, 0].max()], 
                    [y_pert[:, 0].min(), y_pert[:, 0].max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True Marker 1 Values')
    axes[0, 0].set_ylabel('Predicted Marker 1 Values')
    axes[0, 0].set_title(f'Marker 1 Predictions - Perturbations (MAE: {m1_pert_mae:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Marker 1 - Residuals
    residuals_m1_pert = y_pert[:, 0] - m1_pert_pred
    axes[0, 1].scatter(m1_pert_pred, residuals_m1_pert, alpha=0.5, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Marker 1 Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Marker 1 Residual Plot - Perturbations')
    axes[0, 1].grid(True, alpha=0.3)

    # Marker 2 - Scatter plot
    axes[1, 0].scatter(y_pert[:, 1], m2_pert_pred, alpha=0.5, color='green')
    axes[1, 0].plot([y_pert[:, 1].min(), y_pert[:, 1].max()], 
                    [y_pert[:, 1].min(), y_pert[:, 1].max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('True Marker 2 Values')
    axes[1, 0].set_ylabel('Predicted Marker 2 Values')
    axes[1, 0].set_title(f'Marker 2 Predictions - Perturbations (MAE: {m2_pert_mae:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Marker 2 - Residuals
    residuals_m2_pert = y_pert[:, 1] - m2_pert_pred
    axes[1, 1].scatter(m2_pert_pred, residuals_m2_pert, alpha=0.5, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Marker 2 Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Marker 2 Residual Plot - Perturbations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_markers_perturbations_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to '{base_dir}/cellcycle_markers_perturbations_results.png'")
    plt.show()

    # # Distribution plots for perturbations
    # plot_pred_distributions_01(Y_pred_pert, Y_true=Y_true_pert, bins=60, title_prefix="Perturbations - ")
    # print("\nPerturbations distribution plots saved:")
    # print("  - cellcycle_cdt1_distribution.png (updated with perturbations)")
    # print("  - cellcycle_geminin_distribution.png (updated with perturbations)")
    # print("  - cellcycle_joint_distribution.png (updated with perturbations)")
    # print("  - cellcycle_joint_distribution_true.png (updated with perturbations)")

    # 8. Comparative metrics visualization
    print("\n" + "="*60)
    print("Generating comparative metrics plots...")
    print("="*60)

    # Create bar plot comparing Test vs Perturbations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Marker 1 comparison
    datasets = ['Test', 'Perturbations']
    m1_maes = [m1_test_mae, m1_pert_mae]
    bars1 = axes[0].bar(datasets, m1_maes, color=['skyblue', 'orange'], edgecolor='black')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Marker 1 (Cdt1) - MAE Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Marker 2 comparison
    m2_maes = [m2_test_mae, m2_pert_mae]
    bars2 = axes[1].bar(datasets, m2_maes, color=['lightcoral', 'green'], edgecolor='black')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Marker 2 (Geminin) - MAE Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{base_dir}/cellcycle_test_vs_perturbations_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparative plot saved to '{base_dir}/cellcycle_test_vs_perturbations_comparison.png'")
    plt.show()

    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)
