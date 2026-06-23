import os

# Main data/models directory (contains full_cells_fovs/, lion_models_clean/, etc.)
BASE_PATH = '/mnt/new_groups/assafza_group/assafza'
DATA_PATH = '/mnt/new_groups/assafza_group/assafza/full_cells_fovs/train_test_list'

# Local repository/code directory
CWD = '/home/lionb'

model_type = "UNET"
model_path = "./aae_model_ne_27_03_22_128_fl_next"
interpert_model_path = "./unet_model_22_05_22_dna_128b"

patch_size = (32,128,128,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 256
number_epochs = 100
batch_size = 4
batch_norm = True

# input = "channel_target","structure_seg"
input = "channel_signal"
target = "channel_target"

organelle = "Mitochondria"

train_ds_path = ''

test_ds_path = ''

