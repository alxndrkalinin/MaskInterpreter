"""
MaskInterpreter for CIFAR-10 Classification
============================================

This script implements a MaskInterpreter model for generating interpretable
importance masks that explain image classifier predictions. It is designed
for the CIFAR-10 dataset and evaluates robustness on CIFAR-10-C corruptions.

Overview:
---------
The MaskInterpreter learns to generate importance masks that identify which
regions of an input image are critical for the classifier's prediction.
The key insight is that if we perturb non-important regions with noise while
preserving important regions, the classifier's output should remain similar.

Architecture:
-------------
    1. Input Augmentation:
       - Original image (32x32x3)
       - Gradient magnitude channel: computes the gradient of the classifier's
         max predicted probability w.r.t. the input, normalized to [0,1]
       - Augmented input shape: (32x32x4)
    
    2. Generator Network:
       - Conv2D preprocessing layer (32 filters, 3x3)
       - Adaptor network (U-Net) that outputs importance mask (32x32x1)
       - Mask values in [0,1] where 1 = important, 0 = not important
    
    3. Classifier:
       - Pretrained CIFAR-10 classifier (frozen weights)
       - Used to compute similarity between original and adapted predictions

Training Objective:
-------------------
    The model is trained to minimize a composite loss:
    
    1. Similarity Loss (MSE):
       - MSE between classifier outputs on original vs. adapted images
       - Encourages the mask to preserve classifier-relevant information
    
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
    between the classifier's predictions on:
    - The original image
    - The adapted image (important regions preserved, others replaced with noise)
    
    Higher PCC = mask successfully identifies important regions
    Lower PCC = mask fails to capture what the classifier uses

Workflow:
---------
    1. Load pretrained CIFAR-10 classifier
    2. Create MaskInterpreter with U-Net adaptor
    3. Train on CIFAR-10 training data
    4. Evaluate on test set and generate visualizations
    5. Analyze correlation between mask efficacy and classifier accuracy
    6. Find optimal threshold to predict classifier correctness

Output Files:
-------------
    Model:
        - cifar10_mi.h5: Trained MaskInterpreter weights
    
    Visualizations:
        - cifar10_images/*.png: Per-sample visualizations (original, adapted, mask)
        - mask_efficacy_boxplot.png: Boxplot of mask efficacy across datasets
        - accuracy_boxplot.png: Boxplot of classifier accuracy across datasets
        - correlation_mask_efficacy_vs_accuracy.png: Scatter plots per group
        - mask_efficacy_median_barplot.png: Bar plot of median efficacy
        - Mean Mask Efficacy by Dataset Group.png: Bar plot of mean efficacy
        - Prediction of Classifier Correctness using Mask Efficacy Threshold.png

Configuration Flags:
--------------------
    load (bool): If True, load pre-trained MaskInterpreter weights.
                 Default: False
    
    train (bool): If True, train the MaskInterpreter model.
                  Default: True
    
    num_samples (int): Number of samples to visualize.
                       Default: 100
    
    num_samples_subset (int): Number of samples for computing metrics.
                              Default: 500

Key Classes and Functions:
--------------------------
    MaskInterpreter (keras.Model):
        - __init__: Initialize with adaptor network and classifier
        - compile: Set optimizer and loss weights
        - train_step: Custom training loop with gradient augmentation
        - call: Generate importance mask for input images
        - predict_from_noisy: Get classifier prediction on adapted image

"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_addons as tfa  # if you use any tfa functions
from utils.metrics import tf_pearson_corr
# from metrics import tf_pearson_corr
from sklearn.model_selection import train_test_split
from utils.callbacks import *

# Ensure eager execution (TF2 does this by default)
tf.compat.v1.enable_eager_execution()

###############--LOAD--##########################
load = True
train = False
# Number of samples to visualize
num_samples = 100
# We'll use a (small) subset from train, val, test for speed.
num_samples_subset = 500  # adjust as needed
base_dir = os.path.join(gv.CWD, 'cifar10')

class MaskInterpreter(keras.Model):
    def __init__(self, patch_size, adaptor, classifier, weighted_pcc, pcc_target=0.9, **kwargs):
        """
        Args:
            patch_size (tuple): Original input image shape, e.g. (32, 32, 3)
            adaptor (keras.Model): A network that, given processed inputs, outputs a mask.
            classifier (keras.Model): A pretrained classifier for CIFAR-10.
            weighted_pcc (bool): Whether to use weighted Pearson correlation.
            pcc_target (float, optional): Target PCC value between the classifier’s predictions on
                                          the original and perturbed images. Defaults to 0.9.
        """
        super(MaskInterpreter, self).__init__(**kwargs)
        self.weighted_pcc = weighted_pcc
        self.pcc_target = pcc_target
        self.classifier = classifier  # this classifier is assumed fixed
        
        # When augmenting the input with the gradient, the new channel count = original channels + 1.
        augmented_input_shape = (patch_size[0], patch_size[1], patch_size[2] + 1)
        image_input = keras.layers.Input(shape=augmented_input_shape, dtype=tf.float32)
        
        # Process the augmented input.
        processed_image = keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(image_input)
        # Pass through the adaptor network to produce the importance mask.
        # The adaptor is assumed to output a single-channel mask.
        importance_mask = tf.cast(adaptor(processed_image), dtype=tf.float64)
        self.generator = keras.Model(image_input, importance_mask, name="generator")
        
        # ---- Define metrics ----
        self.similiarity_loss_tracker = keras.metrics.Mean(name="similiarity_loss")
        self.binary_size_mask = keras.metrics.BinaryAccuracy(name="binary_size_mask")
        self.importance_mask_size = keras.metrics.Mean(name="importance_mask_size")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.pcc = keras.metrics.Mean(name="pcc")
        self.stop = keras.metrics.Mean(name="stop")
        # (Other heads/metrics can be added as needed.)

    def compile(self, g_optimizer, similiarity_loss_weight=1.0, mask_loss_weight=1.0,
                noise_scale=1.0, target_loss_weight=0.0, run_eagerly=False):
        """
        Args:
            g_optimizer: Optimizer for updating the adaptor network.
            similiarity_loss_weight (float): Weight for the similarity loss.
            mask_loss_weight (float): Weight for the mask loss.
            noise_scale (float): Standard deviation of the noise applied.
            target_loss_weight (float): Weight for the PCC target loss.
            run_eagerly (bool): Whether to run eagerly.
        """
        super(MaskInterpreter, self).compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.similiarity_loss_weight = similiarity_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.noise_scale = noise_scale
        self.target_loss_weight = target_loss_weight

    @property
    def metrics(self):
        return [
            self.similiarity_loss_tracker,
            self.total_loss_tracker,
            self.binary_size_mask,
            self.importance_mask_size,
            self.pcc,
            self.stop,
        ]
    
    def _augment_input_with_gradients(self, x):
        """
        Compute the gradient of the classifier's maximum predicted probability with respect
        to the input x and use its magnitude as an additional channel.
        
        Args:
            x: Tensor of shape (B, H, W, C) (e.g. (B,32,32,3))
        Returns:
            augmented_x: Tensor of shape (B, H, W, C+1)
        """
        # Ensure x is float32 for the classifier.
        x_float = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_float)
            preds = self.classifier(x_float)  # shape (B, num_classes)
            # Use the maximum predicted probability per sample as a scalar.
            max_prob = tf.reduce_max(preds, axis=1)  # shape (B,)
        # Compute gradients of the max probability with respect to x.
        grad = tape.gradient(max_prob, x_float)  # shape (B, H, W, C)
        # Compute the L2 norm of the gradients along the channel dimension.
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=-1, keepdims=True))  # shape (B, H, W, 1)
        # Normalize grad_norm to [0,1] per sample (optional).
        # Here we subtract the min and divide by (max-min) for each sample.
        # For numerical stability, add a small epsilon.
        epsilon = 1e-8
        min_val = tf.reduce_min(grad_norm, axis=[1,2,3], keepdims=True)
        max_val = tf.reduce_max(grad_norm, axis=[1,2,3], keepdims=True)
        grad_norm = (grad_norm - min_val) / (max_val - min_val + epsilon)
        # Concatenate the original input with the grad_norm as an extra channel.
        augmented_x = tf.concat([x, tf.cast(grad_norm, x.dtype)], axis=-1)  # shape (B, H, W, C+1)
        return augmented_x

    def train_step(self, data):
        # Expect data as input images (or (x, y) if needed for other purposes).
        x = tf.cast(data, dtype=tf.float64)  # original input, shape (B, H, W, C)
        # Compute classifier output on the original image (for loss computation).
        classifier_target = tf.cast(self.classifier(x), dtype=tf.float64)
        
        # Augment input with gradient channel.
        # We compute the gradient on the classifier using x (converted to float32 internally).
        augmented_x = self._augment_input_with_gradients(tf.cast(x, tf.float32))
        # Cast augmented input to float32 if necessary.
        augmented_x = tf.cast(augmented_x, tf.float32)
        
        with tf.GradientTape() as tape:
            # Generate importance mask from the augmented input.
            importance_mask = self.generator(augmented_x)  # shape (B, H, W, 1)
            # Create random noise.
            normal_noise = tf.random.normal(tf.shape(importance_mask), stddev=self.noise_scale, dtype=tf.float64)
            # Compute adapted (noisy) image: use importance_mask to mix x and noise.
            adapted_image = (importance_mask * x) + (normal_noise * (1 - importance_mask))
            # Get classifier's output on the adapted image.
            classifier_output = tf.cast(self.classifier(tf.cast(adapted_image, tf.float32)), dtype=tf.float64)
            
            # Compute similarity loss (MSE between classifier outputs on original and adapted images).
            similarity_loss = tf.reduce_mean(tf.square(classifier_target - classifier_output))
            # Regularization on the mask (e.g., encourage sparsity).
            mean_importance_mask = tf.reduce_mean(importance_mask)
            mask_loss = tf.reduce_mean(tf.abs(importance_mask))
            # Compute PCC between classifier outputs.
            pcc_value = tf_pearson_corr(classifier_target, classifier_output)
            pcc_loss = tf.abs(self.pcc_target - pcc_value)
            
            # Total loss.
            total_loss = (similarity_loss * self.similiarity_loss_weight) + \
                         (mask_loss * self.mask_loss_weight) + \
                         (pcc_loss * self.target_loss_weight)
            
        grads = tape.gradient(total_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics.
        self.similiarity_loss_tracker.update_state(similarity_loss)
        self.importance_mask_size.update_state(1 - mean_importance_mask)
        self.binary_size_mask.update_state(tf.zeros_like(importance_mask), importance_mask)
        self.pcc.update_state(pcc_value)
        self.total_loss_tracker.update_state(total_loss)
        self.stop.update_state(pcc_loss + mean_importance_mask)
        
        return {
            "similiarity_loss": self.similiarity_loss_tracker.result(),
            "binary_size": self.binary_size_mask.result(),
            "importance_mask_size": self.importance_mask_size.result(),
            "total_loss": self.total_loss_tracker.result(),
            "pcc": self.pcc.result(),
            "stop": self.stop.result()
        }
    
    def test_step(self, data):
        return self.train_step(data)
    
    def call(self, inputs):
        # For inference, augment the input with gradient channel and produce the mask.
        augmented_x = self._augment_input_with_gradients(tf.cast(inputs, tf.float32))
        return self.generator(tf.cast(augmented_x, tf.float32))
    
    def predict_from_noisy(self, inputs):
        # Given a single image, augment it with gradient channel, compute adapted image and return classifier prediction.
        augmented_x = self._augment_input_with_gradients(tf.cast(np.expand_dims(inputs, axis=0), tf.float32))
        mask = self.generator.predict(augmented_x)
        mask = np.squeeze(mask)  # shape becomes (H,W) or (H,W,1)
        if mask.ndim == 2:
            mask_expanded = np.expand_dims(mask, axis=-1)
        else:
            mask_expanded = mask
        noise = tf.random.normal(shape=inputs.shape, stddev=self.noise_scale, dtype=tf.float64).numpy()
        adapted_image = (mask_expanded * inputs.astype(np.float64)) + ((1 - mask_expanded) * noise)
        pred_probs_adapted = self.classifier.predict(np.expand_dims(adapted_image.astype(np.float32), axis=0))
        return pred_probs_adapted


if __name__ == "__main__":
    # Assume you have a pretrained classifier for CIFAR-10:
    classifier = tf.keras.models.load_model(f'{base_dir}/cifar10_classifier.h5')
    classifier.trainable = False

    # Define a simple adaptor network.
    # For example, a small conv net that outputs a single-channel mask.

    from models.UNETO import *
    adaptor = get_unet((32,32,32),activation="sigmoid") 
    adaptor.summary()

    # Create an instance of the MaskInterpreter.
    mask_interpreter = MaskInterpreter(
        patch_size=(32, 32, 3),
        adaptor=adaptor,
        classifier=classifier,
        weighted_pcc=False,  # or True if tf_pearson_corr supports weighting
        pcc_target=0.95
    )

    # 1. Load and preprocess CIFAR-10 dataset
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Split the original CIFAR-10 training data into a new training set and a validation set (90/10 split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42, shuffle=True
    )
    print("Training set shape:", x_train.shape)
    print("Validation set shape:", x_val.shape)

    # Note: The computed mean and std have shape (1,1,1,3) for broadcasting.
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

    x_train  = (x_train-mean)/std
    x_val  = (x_val-mean)/std
    x_test  = (x_test-mean)/std

    model_path = f"{base_dir}/cifar10_mi.h5"

    # mask_interpreter.build(x_train.shape)
    checkpoint_callback = SaveModelCallback(1,mask_interpreter,model_path,monitor="val_stop",term="val_pcc",term_value=0.92)   

    # 3. Set up EarlyStopping to halt training if validation loss stops improving
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_stop',
        patience=7,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True
    )

    # Compile the MaskInterpreter.
    mask_interpreter.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        
        similiarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    mask_interpreter(np.random.random((1,32,32,3)))

    if load:
        mask_interpreter.load_weights(model_path)

    if train:
        # Suppose you have CIFAR-10 training images in x_train (normalized to [0,1]).
        mask_interpreter.fit(x_train, epochs=200, batch_size=128,callbacks=[early_stop_callback,checkpoint_callback],validation_split=0.1)

        # mask_interpreter.save_weights(model_path)
        
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    import os

    # -----------------------------
    # Assumptions:
    # - x_test: normalized test images, shape (N, 32, 32, 3)
    # - y_test: corresponding labels
    # - x_test_orig: un-normalized test images, computed as (x_test * std) + mean
    # - loaded_classifier: the pretrained classifier loaded from disk
    # - mask_interpreter: the trained MaskInterpreter instance (with its generator available)
    # - mean and std: global mean and standard deviation used for normalization,
    #   with shapes broadcastable to x_test (e.g., (1, 1, 1, 3))
    # -----------------------------


    sample_indices = range(num_samples)#np.random.choice(x_test.shape[0], num_samples, replace=False)
    x_test_orig = (x_test * std) + mean
    x_train_orig = (x_train * std) + mean 
    try:
        os.mkdir(f"{base_dir}/cifar10_images")
    except Exception as e:
        pass
    for idx in sample_indices:
        # Get normalized image and its original version for display
        img_norm = x_train[idx]           # normalized image (32,32,3)
        img_orig = x_train_orig[idx] # un-normalized image for plottin
        true_label = y_train[idx][0] if len(y_train[idx].shape) > 0 else y_train[idx]
        
        # -----------------------------
        # 1. Classifier Prediction on Original Image
        # -----------------------------
        pred_probs_orig = classifier.predict(np.expand_dims(img_norm, axis=0))
        pred_label_orig = np.argmax(pred_probs_orig, axis=-1)[0]
        
        # -----------------------------
        # 2. Generate Importance Mask Using the MaskInterpreter Generator
        # -----------------------------
        # Note: the generator expects a batch, so we add a batch dimension.
        mask = mask_interpreter(np.expand_dims(img_norm, axis=0))
        mask = np.squeeze(mask)  # shape becomes (32,32) or (32,32,1)
        
        # If mask has shape (32,32), expand dims so that it can broadcast with a (32,32,3) image.
        if mask.ndim == 2:
            mask_expanded = np.expand_dims(mask, axis=-1)  # now shape (32,32,1)
        else:
            mask_expanded = mask  # already in (32,32,1)
        
        # -----------------------------
        # 3. Generate a Noisy (Adapted) Image Using the Importance Mask
        # -----------------------------
        # Generate random noise of the same shape as the image (in normalized domain)
        # Use the noise scale from the mask_interpreter.
        noise = tf.random.normal(shape=img_norm.shape, 
                                stddev=mask_interpreter.noise_scale, 
                                dtype=tf.float64)
        noise = noise.numpy()  # convert tensor to numpy array
        
        # noise = np.zeros_like(img_norm.shape)

        # Compute the adapted image: use the mask to blend the original image and noise.
        # Areas with high importance (mask ~ 1) keep the original image,
        # areas with low importance (mask ~ 0) are replaced by noise.
        adapted_image = (mask_expanded * img_norm.astype(np.float64)) + ((1 - mask_expanded) * noise)
        
        # -----------------------------
        # 4. Classifier Prediction on Adapted (Noisy) Image
        # -----------------------------
        pred_probs_adapted = classifier.predict(
            np.expand_dims(adapted_image.astype(np.float32), axis=0)
        )
        pred_label_adapted = np.argmax(pred_probs_adapted, axis=-1)[0]
        
        # -----------------------------
        # 5. Revert Normalization for Display
        # -----------------------------
        # For both the original image and the adapted image, we revert normalization:
        # original = (normalized * std) + mean
        adapted_image_orig = (adapted_image * std) + mean
        adapted_image_orig = np.clip(adapted_image_orig, 0, 1)[0]
        mask_efficacy = tf_pearson_corr(pred_probs_adapted,pred_probs_orig)
        # -----------------------------
        # 6. Plot the Results
        # -----------------------------
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Original image with true and predicted labels
        plt.subplot(1, 3, 1)
        plt.imshow(img_orig)
        plt.title(f"Original\nTrue: {true_label}, Pred: {pred_label_orig}")
        plt.axis('off')
        
        # Subplot 2: Adapted (noisy) image with classifier prediction on it
        plt.subplot(1, 3, 2)
        plt.imshow(adapted_image_orig)
        plt.title(f"Adapted (Noisy)\nPred: {pred_label_adapted}")
        plt.axis('off')
        
        # Subplot 3: Original image with the importance mask overlaid as a heatmap
        plt.subplot(1, 3, 3)
        plt.imshow(img_orig)
        plt.imshow(mask, cmap='jet', alpha=0.5)
        plt.title("Mask Overlay: {}".format(mask_efficacy))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{base_dir}/cifar10_images/{idx}.png")
        
        
    import os
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # =============================================================================
    # Helper: Compute mask efficacy for a given image using predict_from_noisy
    # =============================================================================
    def get_mask_efficacy(img, mask_interpreter):
        """
        Given a normalized image (e.g. shape (32,32,3)), use the MaskInterpreter to
        get predictions on the original and adapted (noisy) image, and return the PCC.
        """
        # The predict_from_noisy method returns (original_pred, noisy_pred, pcc_value)
        noisy_prediction_probs = mask_interpreter.predict_from_noisy(img)
        prediction_probs = mask_interpreter.classifier.predict(np.expand_dims(img,axis=0))
        pcc_value = tf_pearson_corr(noisy_prediction_probs,prediction_probs)
        return pcc_value


    def compute_accuracy(img, true_label, classifier):
        """
        Compute a continuous measure of accuracy: the classifier's predicted probability 
        for the true class.
        """
        pred = classifier.predict(np.expand_dims(img, axis=0))
        # true_label may be an array; ensure it's an integer.
        
        true_label = int(true_label)
        # return pred[0, true_label]
        
        if np.argmax(pred[0])==true_label:
            return 1
        else:
            return 0


    # ------------------------------------------------------------------
    # Define the groups for which we want metrics:
    # Groups: 'train', 'val', 'test'
    # ------------------------------------------------------------------
    mask_efficacy_dict = {}
    accuracy_dict = {}

    # ---- For training data ----
    eff_train = []
    acc_train = []
    print("Train...")
    for i in tqdm(range(min(num_samples_subset, x_train.shape[0]))):
        img = x_train[i]
        # y_train is typically shape (N,1); extract integer label.
        lbl = y_train[i][0] if len(y_train[i].shape) > 0 else y_train[i]
        eff_train.append(get_mask_efficacy(img, mask_interpreter))
        acc_train.append(compute_accuracy(img, lbl, mask_interpreter.classifier))
    mask_efficacy_dict['train'] = np.array(eff_train)
    accuracy_dict['train'] = np.array(acc_train)

    # ---- For validation data ----
    eff_val = []
    acc_val = []
    print("Val...")
    for i in tqdm(range(min(num_samples_subset, x_val.shape[0]))):
        img = x_val[i]
        lbl = y_val[i][0] if len(y_val[i].shape) > 0 else y_val[i]
        eff_val.append(get_mask_efficacy(img, mask_interpreter))
        acc_val.append(compute_accuracy(img, lbl, mask_interpreter.classifier))
    mask_efficacy_dict['val'] = np.array(eff_val)
    accuracy_dict['val'] = np.array(acc_val)

    # ---- For test data ----
    eff_test = []
    acc_test = []
    print("Test...")
    for i in tqdm(range(min(num_samples_subset, x_test.shape[0]))):
        img = x_test[i]
        lbl = y_test[i][0] if len(y_test[i].shape) > 0 else y_test[i]
        eff_test.append(get_mask_efficacy(img, mask_interpreter))
        acc_test.append(compute_accuracy(img, lbl, classifier))
    mask_efficacy_dict['test'] = np.array(eff_test)
    accuracy_dict['test'] = np.array(acc_test)
    
    # ------------------------------------------------------------------
    # Plotting: Create two boxplots (one for mask efficacy, one for accuracy).
    # ------------------------------------------------------------------

    # Order the groups: train, val, test, then sorted corruption names.
    group_names = ['train', 'val', 'test']
        
    # Prepare data lists in that order.
    efficacy_data = [mask_efficacy_dict[k] for k in group_names]
    accuracy_data = [accuracy_dict[k] for k in group_names]

    # ---- Boxplot for Mask Efficacy ----
    plt.figure(figsize=(14, 6))
    plt.boxplot(efficacy_data, labels=group_names, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mask Efficacy (PCC)")
    plt.title("Boxplot of Mask Efficacy (PCC between original & noisy predictions)")
    plt.tight_layout()
    plt.savefig("mask_efficacy_boxplot.png")
    plt.show()

    # ---- Boxplot for Accuracy (True Class Probability) ----
    plt.figure(figsize=(14, 6))
    plt.boxplot(accuracy_data, labels=group_names, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Classifier Confidence for True Class")
    plt.title("Boxplot of Classifier Accuracy (Probability for True Class)")
    plt.tight_layout()
    plt.savefig("accuracy_boxplot.png")


    import scipy.stats
import global_vars as gv

    # Example: Assuming these dictionaries exist.
    # mask_efficacy_dict = { 'train': np.array([...]), 'val': np.array([...]), ... }
    # accuracy_dict = { 'train': np.array([...]), 'val': np.array([...]), ... }

    # Define the group ordering.
    # Ensure that groups for train, val, and test come first, then the corruption groups sorted alphabetically.
    group_names = ['train', 'val', 'test']

    num_groups = len(group_names)
    cols = 3  # number of subplots per row; adjust as needed.
    rows = int(np.ceil(num_groups / cols))

    plt.figure(figsize=(5 * cols, 5 * rows))

    # Loop over each group to create a subplot.
    for i, group in enumerate(group_names):
        # Extract data for this group.
        efficacy = mask_efficacy_dict[group]  # mask efficacy (PCC) values for each sample.
        accuracy = accuracy_dict[group]       # classifier's true class probability for each sample.
        
        # Compute Pearson correlation between mask efficacy and accuracy.
        r, p_val = scipy.stats.pearsonr(efficacy, accuracy)
        
        # Create a scatter plot.
        ax = plt.subplot(rows, cols, i + 1)
        ax.scatter(efficacy, accuracy, alpha=0.5, label='Samples')
        
        # Optionally, compute and overlay a regression (best–fit) line.
        slope, intercept, r_value, p_value_lin, std_err = scipy.stats.linregress(efficacy, accuracy)
        x_fit = np.linspace(np.min(efficacy), np.max(efficacy), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='red', label='Fit line')
        
        # Annotate the plot with the correlation coefficient and p–value.
        ax.set_title(f"{group}\nr = {r:.2f}, p = {p_val:.2e}")
        ax.set_xlabel("Mask Efficacy (PCC)")
        ax.set_ylabel("Accuracy (True Class Confidence)")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{base_dir}/correlation_mask_efficacy_vs_accuracy.png")
    plt.show()


    # Assuming group_names is defined as follows:
    group_names = ['train', 'val', 'test']

    # Compute the mean mask efficacy for each group.
    mean_efficacies = [np.median(mask_efficacy_dict[group]) for group in group_names]

    # Create a bar plot.
    plt.figure(figsize=(12, 6))
    bars = plt.bar(group_names, mean_efficacies, color='skyblue', edgecolor='black')

    # Annotate each bar with its mean value.
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.xlabel("Dataset Group")
    plt.ylabel("Median Mask Efficacy (PCC)")
    plt.title("Median Mask Efficacy by Dataset Group")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/mask_efficacy_median_barplot.png")
    plt.show()