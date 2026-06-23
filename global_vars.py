import os

# Main data/models directory
BASE_PATH = 'path/to/base/directory'  # Replace with your base directory path
DATA_PATH = 'path/to/train_test_list'  # Replace with the path to your train/test list

# Local repository/code directory
CWD = 'path/to/current/working/directory'  # Replace with your current working directory

model_type = "UNET" #MG for running mask interpreter UNET for running in silico labeling model
model_path = "path/to/model"  # Replace with the path to your model or where to save model upon training
interpert_model_path = "path/to/interpert_model"  # Replace with the path to model you want to interpret

patch_size = (32,128,128,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 256
number_epochs = 100
batch_size = 4
batch_norm = True

input = "channel_signal"
target = "channel_target"

organelle = "Mitochondria"

train_ds_path = ''

test_ds_path = ''

