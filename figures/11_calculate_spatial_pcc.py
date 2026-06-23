import sys
import os

# Add parent directory to path to import cell_generator modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import tensorflow.keras as keras
from utils.metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
from utils.utils import *
import cv2
from dataset import DataGen
import matplotlib.pyplot as plt
import matplotlib.cm as cm


params = [
        #   {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5},
        #   {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5},
        #   {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5},
        #   {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5},
        #   {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0},
        #   {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5},
        #   {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5},
        # {"organelle":"DNA","model":"../mg_model_dna_13_05_24_1.5b","noise":1.5},
          ]

gv.input = "channel_signal"
gv.target = "channel_target"
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
def auto_balance(image):
    """Auto balance the image similar to ImageJ's Auto Contrast function."""
    image = image.astype(np.float32)
    plow, phigh = np.percentile(image, (0.1, 99.9))
    image = np.clip((image - plow) / (phigh - plow), 0, 1)
    return image

def calculate_spatial_pcc(image1,image2):
    # Compute the Pearson correlation coefficient matrix for each slice of the 3D images
    corr_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    occur_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    for z in range(0,image1.shape[0]- z_step,z_step):
        for i in range(0,image1.shape[1]- xy_step,xy_step):
            for j in range(0,image1.shape[2] - xy_step,xy_step):
                # corr_matrix[z:z+z_step,i:i+xy_step, j:j+xy_step] += np.corrcoef(image1[z:z+z_step,i:i+xy_step, j:j+xy_step].flatten(), image2[z:z+z_step,i:i+xy_step, j:j+xy_step].flatten())[0, 1]
                corr_matrix[z:z+z_step,i:i+xy_step, j:j+xy_step] +=tf_pearson_corr(image1[z:z+z_step,i:i+xy_step, j:j+xy_step], image2[z:z+z_step,i:i+xy_step, j:j+xy_step])[0, 1]
                occur_matrix[z:z+z_step,i:i+xy_step, j:j+xy_step] +=1.0
    return (corr_matrix).astype(np.float16), occur_matrix.astype(np.float16)           

def predict_images_and_calculate_spatial_pcc(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False,noise_scale=1.5):
    """This method run analysis that find the std of the noise that need to be used for the input data
        We test the prediction of noisy input by adding noise to the input with different stds, the noisy input is 0.5 signal, 0.5 noise.
    Args:
        dataset (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to gv.model_path.
        model (_type_, optional): _description_. Defaults to None.
        images (_type_, optional): _description_. Defaults to range(10).
        weighted_pcc (bool, optional): _description_. Defaults to False.
        #######TODO: need to change docs
    """
    batch_size=4
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)

    count=0
    for image_index in images:
        dir_path = "{}/spatial_pcc/{}".format(model_path,image_index)
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
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, 64, 16)
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
        unet_patchs_p = predict(model.unet,input_patchs,batch_size,gv.patch_size)
        mask_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
        
        ## Back to image 
        unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape,gv.patch_size,64,16)
        mask_p = mask_p_full/d
        
        ## Create noise vector
        normal_noise = tf.random.normal(tf.shape(input_patchs),stddev=noise_scale,dtype=tf.float64)
        ## Create noisy input and predict unet
        mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p, gv.patch_size, 64, 16)
        mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
        input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
        unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size,gv.patch_size)

        ## Back to image 
        input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,64,16)
        ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(dir_path,image_index)) 
        ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/unet_noise_prediction_{}.tiff".format(dir_path,image_index))
        ImageUtils.imsave((target_image).astype(np.float16),"{}/target_{}.tiff".format(dir_path,image_index)) 
        
        del unet_patchs_p
        del input_patchs_p      
        del unet_noise_patchs_p
        del input_p
        del normal_noise
        del input_patchs
        del mask_patchs_p_term
        del mask_noise_patchs
        
        prediction_to_gt_spatial_pcc, occur_matrix = calculate_spatial_pcc(target_image,unet_p/d)
        prediction_to_gt_spatial_pcc = np.expand_dims(prediction_to_gt_spatial_pcc/d, axis=-1)
        ImageUtils.imsave(prediction_to_gt_spatial_pcc,"{}/prediction_to_gt_spatial_pcc_{}.tiff".format(dir_path,image_index))
        
        # Create overlay image: prediction with spatial PCC heatmap
        th=0.0
        prediction_normalized = unet_p/d
        pcc_for_overlay = prediction_to_gt_spatial_pcc[:,:,:,0]
        # pcc_normalized = np.where(1-np.abs(pcc_for_overlay)<th,1-np.abs(pcc_for_overlay),0.0)#
        # pcc_normalized = 1 - np.clip(pcc_for_overlay, 0, 1)  # move range [-1,1] to [0,1]
        pcc_normalized = pcc_for_overlay
        
        # Create overlay for each z-slice
        overlay_volume = []
        for z in range(prediction_normalized.shape[0]):
            pred_slice = auto_balance(prediction_normalized[z, :, :, 0].astype(np.float32))
            pcc_slice = pcc_normalized[z, :, :].astype(np.float32)
            
            # Convert prediction to RGB (grayscale in all channels)
            pred_rgb = np.stack([pred_slice, pred_slice, pred_slice], axis=-1)
            
            # Apply red colormap to PCC values (higher correlation = brighter red)
            pcc_colored = np.stack([pcc_slice, np.zeros_like(pcc_slice), np.zeros_like(pcc_slice)], axis=-1)
            
            # Blend prediction and PCC heatmap (25% prediction, 75% heatmap)
            overlay_slice = 0.7 * pred_rgb + 0.3 * pcc_colored
            
            overlay_volume.append(overlay_slice)
        
        overlay_volume = np.stack(overlay_volume, axis=0).astype(np.float32)
        ImageUtils.imsave(overlay_volume, "{}/prediction_with_spatial_pcc_overlay_{}.tiff".format(dir_path,image_index))
        
        # Save slice as JPG
        slice_idx = 24#overlay_volume.shape[0] // 2
        
        # Create custom red colormap to match the PCC overlay
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colors as mcolors
        colors = [(0, 0, 0), (1, 0, 0)]  # Black to Red
        n_bins = 16
        red_cmap = LinearSegmentedColormap.from_list('red', colors, N=n_bins)
        
        fig, ax = plt.subplots(figsize=(40, 10))
        ax.imshow(overlay_volume[slice_idx])
        ax.axis('off')
        
        # Create colorbar using actual PCC values from prediction_to_gt_spatial_pcc
        norm = mcolors.Normalize(vmin=pcc_normalized.min(), vmax=pcc_normalized.max())
        sm = cm.ScalarMappable(cmap=red_cmap, norm=norm)
        sm.set_array([])
        
        # Add colorbar on the left side with same height as image
        cbar = fig.colorbar(sm, ax=ax, location='left', fraction=0.046, pad=0.04, shrink=0.9, aspect=20)
        cbar.set_label('PCC (Pearson Correlation)', rotation=90, labelpad=15)
        
        plt.tight_layout(pad=0)
        plt.savefig("{}/prediction_with_spatial_pcc_overlay_{}_slice{}.jpg".format(dir_path, image_index, slice_idx), 
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        # prediction_to_noisy_spatial_pcc = calculate_spatial_pcc(unet_noise_p/d,unet_p/d)
        # prediction_to_noisy_spatial_pcc = np.expand_dims(prediction_to_noisy_spatial_pcc, axis=-1)
        # ImageUtils.imsave(prediction_to_noisy_spatial_pcc,"{}/prediction_to_noisy_spatial_pcc_{}.tiff".format(dir_path,image_index))
        
        # spatial_pcc_for_prediction = calculate_spatial_pcc(prediction_to_gt_spatial_pcc,prediction_to_noisy_spatial_pcc)
        # spatial_pcc_for_prediction = np.expand_dims(spatial_pcc_for_prediction, axis=-1)
        # ImageUtils.imsave(spatial_pcc_for_prediction,"{}/prediction_to_gt_to_prediction_to_noisy_spatial_pcc{}.tiff".format(dir_path,image_index))
        
        # pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:],target_seg_image_dilated)
        # pcc_gt = pearson_corr((unet_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
        # score_for_mask_efficacy_in_pcc = pearson_corr(prediction_to_gt_spatial_pcc,prediction_to_noisy_spatial_pcc)
        # print("pearson corr for with noisy image:{} is :{}".format(image_index,pcc))
        # print("pearson corr for gt image:{} is :{}".format(image_index,pcc_gt))
        # print("pearson corr for both spatial pccs image:{} is :{}".format(image_index,score_for_mask_efficacy_in_pcc))
        
        del unet_p
        del unet_noise_p
        

for param in params:
    try:
        print(param["model"])
        base_path = os.path.join(gv.DATA_PATH, "spatial_pcc/{}".format(param["model"].split('/')[-1]))
        create_dir_if_not_exist(base_path)
        # ds_path = "{}/merged_dataset.csv".format(base_path)
        ds_path = os.path.join(gv.DATA_PATH, f'{param["organelle"]}/image_list_test.csv')
        dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False,image_path_col='path_tiff')
        print("# images in dataset:",dataset.df.data.shape[0])  
        predict_images_and_calculate_spatial_pcc(dataset,model_path=param["model"],images=[1],weighted_pcc=False,noise_scale=param["noise"])            
    except Exception as e:
        print(e)

#range(0,min(10,dataset.df.data.shape[0]),1)