import os
import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import global_vars as gv

load = False
base_dir = os.path.join(gv.CWD, 'cifar10')
# 1. Load and preprocess CIFAR-10 dataset
print("Loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Compute the global mean and standard deviation over the training set.
# Note: The computed mean and std have shape (1,1,1,3) for broadcasting.
mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

# Normalize (scale) the training and test images.
x_train = (x_train - mean) / std
x_test  = (x_test - mean) / std

# Split the original CIFAR-10 training data into a new training set and a validation set (90/10 split)
x_train_new, x_val, y_train_new, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, shuffle=True
)
print("Training set shape:", x_train_new.shape)
print("Validation set shape:", x_val.shape)

# 2. Build a simple CNN model
print("\nBuilding model...")
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
#                            input_shape=(32, 32, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
    
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# Load pre-trained VGG19 model
base_model = tf.keras.applications.VGG19(
    weights="imagenet",  # Load weights pre-trained on ImageNet
    include_top=False,   # Exclude the fully connected layers
    input_shape=(32, 32, 3)  # CIFAR-10 input size
)

# Freeze base model layers
# base_model.trainable = False  # Optional: Set to True for fine-tuning

# Add custom classification head
inputs = tf.keras.Input(shape=(32, 32, 3))
x = base_model(inputs, training=True)  # Prevent batch norm issues
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)

# Create the model
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3. Set up EarlyStopping to halt training if validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)

save_path = f"{base_dir}/cifar10_classifier.h5"

if not load:
    # 4. Train the model using the training set and validate on the validation set
    print("\nTraining model...")
    history = model.fit(
        x_train_new, y_train_new,
        epochs=50,              # High maximum epoch count; training will stop early if no improvement
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )
    model.save(save_path)
else:
    model = tf.keras.models.load_model(save_path)


print(f"Classifier saved to {save_path}")

# Evaluate on the training set
print("\nEvaluating on CIFAR-10 training set...")
train_loss, train_acc = model.evaluate(x_train_new, y_train_new, verbose=0)
print("CIFAR-10 Train Accuracy: {:.4f}".format(train_acc))

# Evaluate on the validation set
print("\nEvaluating on CIFAR-10 validation set...")
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print("CIFAR-10 Validation Accuracy: {:.4f}".format(val_acc))

# Evaluate on the test set (which has not been used for training/validation)
print("\nEvaluating on CIFAR-10 test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("CIFAR-10 Test Accuracy: {:.4f}".format(test_acc))

