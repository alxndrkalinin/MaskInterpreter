import tensorflow as tf
import tensorflow.keras as keras
from utils.metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
from utils.utils import *
import os
import cv2
from dataset import DataGen
import pandas as pd
import glob


params = [
          {"organelle":"Plasma-membrane","model":"../unet_model_22_05_22_membrane_128","target":"channel_membrane"},
          {"organelle":"DNA","model":"../unet_model_22_05_22_dna_128","target":"channel_dna"}
          ]

gv.input = "channel_signal"
weighted_pcc = False


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


## Patch steps
xy_step = 64
z_step = 16

batch_size = 4    
path = os.path.join(gv.DATA_PATH, '**/image_list_with_metadata__with_efficacy_scores_full.csv')

# Print the path pattern
print(f"Using path pattern: {path}")

# Use glob to get all the CSV file paths
csv_files = glob.glob(path, recursive=True)

# Print the found files
print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(file)

# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Check if any DataFrames were created
if not dfs:
    print("No DataFrames were created. Please check the CSV files.")
else:
    # Concatenate all DataFrames
    metadata_with_efficacy_scores_df = pd.concat(dfs, ignore_index=True)
    print("metadata_with_efficacy_scores_df # FOVS:{}".format(metadata_with_efficacy_scores_df.shape[0]))
    base_dir = os.path.join(gv.BASE_PATH, 'full_cells_fovs/train_test_list/unet_predictions')
    create_dir_if_not_exist(base_dir)
    ds_path = "{}/metadata_with_efficacy_scores_and_unet_scores.csv".format(base_dir)
    metadata_with_efficacy_scores_df.to_csv(ds_path)
    
def predict_images_unet(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False):
    """
    Args:
        dataset (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to gv.model_path.
        model (_type_, optional): _description_. Defaults to None.
        images (_type_, optional): _description_. Defaults to range(10).
        weighted_pcc (bool, optional): _description_. Defaults to False.
        #######TODO: need to change docs
    """
    pccs = []
    batch_size=4
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)

    count=0
    for image_index in images:
        dir_path = "{}/unet_predictions/{}".format(model_path,image_index)
        create_dir_if_not_exist(dir_path)
        count+=1
        print("image index: {}/{}".format(count,len(images)))
        
        slice_by = None
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        px_start=0
        py_start=0
        pz_start = 0
        px_end=input_image.shape[1]
        py_end=input_image.shape[2]
        if target_seg_image is not None:
            target_seg_image = target_seg_image/255.
        else:
            target_seg_image = np.zeros_like(target_image)
            
        if mem_seg_image is not None:
            mem_seg_image = mem_seg_image/255.
        else:
            mem_seg_image = np.ones_like(target_image)
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
        if weighted_pcc:
            target_seg_image_dilated = np.copy(target_seg_image)
            for h in range(target_seg_image.shape[1]):
                target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((25,25)))  
        else:
            target_seg_image_dilated = None
        
        weights = get_weights(input_patchs[0].shape)    
        
        ## Batch predict
        ## Predicte unet and mask
        
        print("batch predict...")
        unet_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
        
        ## Back to image 
        unet_p,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,np.ones_like(unet_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)

        ## Back to image 
        # ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(dir_path,image_index)) 
        # ImageUtils.imsave((target_image).astype(np.float16),"{}/target_{}.tiff".format(dir_path,image_index)) 
        
        del unet_patchs_p
        del input_patchs
        
        pcc_gt = pearson_corr((unet_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
        print("pearson corr for gt image:{} is :{}".format(image_index,pcc_gt))
        pccs.append(pcc_gt)
        del unet_p
    return pccs
        
for param in params:
    try:
        print(param["model"])
        gv.target = param["target"]
        dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False,image_path_col='combined_image_storage_path')
        print("# images in dataset:",dataset.df.data.shape[0])
        pccs = predict_images_unet(dataset,model_path=param["model"],images=range(dataset.df.data.shape[0]),weighted_pcc=False)
        metadata_with_efficacy_scores_df[param["model"]] = pccs
    except Exception as e:
        print(e)
        
metadata_with_efficacy_scores_df.to_csv(ds_path)
