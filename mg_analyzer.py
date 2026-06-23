"""
Mask Interpreter Analyzer
=======================

This script provides comprehensive analysis tools for evaluating MaskInterpreter
models on 3D microscopy images. It computes mask efficacy, tests noise robustness,
and analyzes the relationship between importance masks and organelle segmentations.

Overview:
---------
The analyzer works with trained MaskInterpreter models to:
1. Compute U-Net prediction quality (PCC with ground truth)
2. Find optimal noise scale for perturbation experiments
3. Analyze importance masks at different thresholds
4. Evaluate correlations between mask regions and organelle locations
5. Generate predictions and save analysis results

Key Concepts:
-------------
    Mask Efficacy:
        Measured as Pearson Correlation Coefficient (PCC) between U-Net predictions
        on original images vs. noisy/masked images. Higher PCC = mask preserves
        important information.
    
    Threshold Analysis:
        Binary thresholding of importance masks at various levels (0.0-1.0) to
        determine optimal threshold for identifying important regions.
    
    Noise Perturbation:
        Adding Gaussian noise to non-important regions (where mask < threshold)
        to test if the mask correctly identifies what the model uses.

Analysis Functions:
-------------------
    calc_unet_pcc(dataset, model_path, model, images, weighted_pcc):
        Calculate U-Net model performance by computing PCC between predictions
        and ground truth target images.
        
        Output: pcc_results.csv with PCC values per image
    
    find_noise_scale(dataset, model_path, model, images, weighted_pcc):
        Find optimal noise standard deviation for perturbation experiments.
        Tests noise STDs from 0.0 to 4.5 in 0.5 steps.
        Creates noisy inputs: original + noise * std
        
        Output: noise_pcc_results.csv with PCC for each noise level
    
    analyze_th(dataset, mode, mask_image, manual_th, save_image, ...):
        Main analysis function - generates masks and predictions using
        different thresholding strategies.
        
        Modes:
            - "agg": Aggregate mode - threshold all values below TH (keep high importance)
            - "loo": Leave-one-out - only keep values in current TH bin
            - "mask": Use external mask (for GradCAM, guided backprop comparison)
            - "regular": Use importance mask as-is (optionally with manual TH)
        
        Output:
            - pcc_results.csv: PCC between original and noisy predictions
            - mask_size_results.csv: Mask size at each threshold
            - context_results.csv: Intersection of mask with organelle segmentation
            - Optional: Image visualizations and histograms
    
    

Patch-Based Processing:
-----------------------
    Images are processed in patches due to memory constraints:
    
    1. collect_patchs(): Extract overlapping patches from full image
    2. predict(): Batch predict on all patches
    3. assemble_image(): Reconstruct full image from patch predictions
                         using weighted averaging for overlap regions

Configuration:
--------------
    Image Center and Margins:
        center_xy = [312, 462]    # Center point for cropping
        margin = [192, 256]       # Margins around center
    
    Patch Parameters:
        xy_step = 64              # Horizontal/vertical step between patches
        z_step = 16               # Z-axis step between patches
        batch_size = 4            # Batch size for prediction
    
    Threshold Parameters:
        ths_start = 0.0           # Starting threshold
        ths_step = 0.1            # Threshold increment
        ths_stop = 1.0            # Ending threshold (exclusive)

Output Files:
-------------
    Results CSVs:
        - pcc_results.csv: Pearson correlation coefficients
        - mask_size_results.csv: Size of thresholded masks
        - context_results.csv: Mask-organelle intersection metrics
        - noise_pcc_results.csv: PCC at different noise levels
        - corr_results_*.csv: Correlation analysis results
        - predictions_results_*.csv: Prediction quality metrics
    
    Images (TIFF format):
        - input_*.tiff: Original input images
        - target_*.tiff: Ground truth target images
        - unet_prediction_*.tiff: U-Net model predictions
        - mask_*.tiff: Importance masks
        - noisy_input_*.tiff: Input with noise applied
        - noisy_unet_prediction_*.tiff: Prediction on noisy input
        - seg_target_*.tiff: Organelle segmentation

Metrics Computed:
-----------------
    - PCC: Pearson Correlation Coefficient between predictions
    - Mask Size: Fraction of image covered by mask (above threshold)
    - Organelle Intersection: Overlap between mask and organelle segmentation
    - Context: Ratio of organelle pixels in mask region

Weighted PCC:
-------------
    When weighted_pcc=True, the PCC is computed only within a dilated
    region around the target organelle segmentation. This focuses the
    metric on relevant image regions.

Data Format:
------------
    Input Images: 3D microscopy volumes (Z, Y, X) or (Z, Y, X, C)
    Patch Size: Defined in global_vars.patch_size
    Normalization: Per-image normalization applied in preprocess_image()

Dependencies:
-------------
    - tensorflow / keras
    - numpy, opencv (cv2)
    - matplotlib
    - cell_imaging_utils (custom imaging utilities)
    - Custom modules: dataset (DataGen), metrics, utils, global_vars

Usage Examples:
---------------
    # Calculate U-Net performance
    calc_unet_pcc(dataset, model_path="./my_model", images=range(10))
    
    # Find optimal noise scale
    find_noise_scale(dataset, model_path="./my_model", images=range(10))
    
    # Analyze thresholds in aggregate mode
    analyze_th(dataset, mode="agg", save_image=True, weighted_pcc=True)
    
    # Analyze with manual threshold
    analyze_th(dataset, mode="regular", manual_th=0.85, save_image=5)
    
    # Evaluate organelle correlations
    analyze_correlations_constant(
        organelles=["Mitochondria", "Golgi"],
        mask_th=0.7,
        absoulte_organelle_precent_pixels=0.05
    )
"""

import gc
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from utils.metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
import global_vars as gv
from utils.utils import *
import os
import matplotlib.pyplot as plt
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

## Image center
center_xy = [312,462] #[200,100]
margin=[192,256] #[256,256]#

## Patch steps
xy_step = 64
z_step = 16

batch_size = 4

weighted_pcc = False

def calc_unet_pcc(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False):
    """This method run analysis to calc unet model perfermence
    Args:
        dataset (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to gv.model_path.
        model (_type_, optional): _description_. Defaults to None.
        images (_type_, optional): _description_. Defaults to range(10).
        weighted_pcc (bool, optional): _description_. Defaults to False.
    """
    batch_size=4
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)
        
    dir_path = "{}".format(model_path)
    create_dir_if_not_exist(dir_path)
    
    pcc_results = DatasetMetadataSCV("{}/pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(["PCC"])

    count=0
    for image_index in images:
        count+=1
        print("image index: {}/{}".format(count,len(images)))
        pccs = []
        
        slice_by = None
        # slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        if slice_by is None:
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
        unet_p,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,np.ones_like(input_patchs)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        
        pcc_gt = pearson_corr((target_image), (unet_p/d),target_seg_image_dilated)
        print("pearson with gt corr for image:{} is :{}".format(image_index,pcc_gt))
        
        del unet_patchs_p
        
        pcc_results.add_row(pcc_gt)
        
        del input_patchs
        
    pcc_results.create()

def find_noise_scale(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False):
    """This method run analysis that find the std of the noise that need to be used for the input data
        We test the prediction of noisy input by adding noise to the input with different stds, the noisy input is 0.5 signal, 0.5 noise.
    Args:
        dataset (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to gv.model_path.
        model (_type_, optional): _description_. Defaults to None.
        images (_type_, optional): _description_. Defaults to range(10).
        weighted_pcc (bool, optional): _description_. Defaults to False.
    """
    batch_size=4
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)
        
    ## Create noise vars
    noise_start = 0.0
    noise_step = 0.5
    noise_stop = 4.5
    noises = np.arange(noise_start,noise_stop,noise_step)
        
    dir_path = "{}".format(model_path)
    create_dir_if_not_exist(dir_path)
    
    # results of pcc for the prediction from original and noisy input by noise std
    pcc_results = DatasetMetadataSCV("{}/noise_pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(noises)

    count=0
    for image_index in images:
        count+=1
        print("image index: {}/{}".format(count,len(images)))
        pccs = []
        
        ## Preprocess images
        # px_start=center_xy[0]-margin[0]-xy_step
        # py_start=center_xy[1]-margin[1]-xy_step
        # pz_start = 0
        # px_end=center_xy[0]+margin[0]+xy_step
        # py_end=center_xy[1]+margin[1]+xy_step
        
        slice_by = None
        # slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        if slice_by is None:
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
        unet_p,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,np.ones_like(input_patchs)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        
        pcc_gt = pearson_corr((target_image), (unet_p/d),target_seg_image_dilated)
        print("pearson with gt corr for image:{} is :{}".format(image_index,pcc_gt))
        
        del unet_patchs_p
        
        for noise_scale in noises:
            print(noise_scale)
            ## Create noise vector
            normal_noise = tf.random.normal(tf.shape(input_patchs),stddev=noise_scale,dtype=tf.float64)
            
            ## Create noisy input and predict unet
            input_patchs_p = input_patchs+normal_noise
            unet_noise_patchs_p = predict(model,input_patchs_p.numpy(),batch_size,gv.patch_size)
    
            ## Back to image 
            input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
            
            pcc = pearson_corr((unet_p/d), (unet_noise_p/d),target_seg_image_dilated)
            
            pccs.append(pcc)
            print("pearson corr for image:{} with noise std:{}, is :{}".format(image_index,noise_scale,pcc))
            
            del input_patchs_p      
            del unet_noise_patchs_p
            del input_p
            del unet_noise_p
            del normal_noise
            
        pcc_results.add_row(pccs)
        
        del input_patchs
        
    pcc_results.create()

def analyze_th(dataset,mode,mask_image=None,manual_th="full",save_image=True,save_histo=False,weighted_pcc = False, model_path=gv.model_path,model=None,compound=None,images=range(10), noise_scale = 1.5,save_results=True,results_save_path=None):
    """method that generate masks and predictions by thresholding the importance mask with different TH

    Args:
        dataset (DataGen): instance that point to the dataset, pertrub or regular according to the organelle.
        mode (str): "agg" - th all the values below the th
                    "loo" - th all the values except the current th
                    "mask" - by external mask - used to evaluate gradcam, guided back propagation and manual validations 
                    "regular" - use the importance mask as is without binary th, if manual_th is not "full" then use the manual_th value
        mask_image (np ndarray, optional): if mode == "mask" then this is the mask to use. Defaults to None. values are 0 or 255
        mask_mode_pred_path (str): if mode == "mask" then this is the path for data to be saved
        manual_th (str, float optional): _description_. Defaults to "full".
        save_image (bool, int optional): if True save all images, if False save nothing, if int, save the first images according to the int. Defaults to True.
        save_histo (bool, optional): Save histograms of the importance masks. Defaults to False.
        weighted_pcc (bool, optional): use regular PCC or weighted_PCC. Defaults to False.
        model_path (str, optional): use to load model if not passed and dir to save results. Defaults to gv.model_path.
        model (model, optional): model if exists, if None method will load model from model_path. Defaults to None.
    """
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)
        
    ## Create thresholds
    num_bins = 10 # number of bins for the histogram
    ths_start = 0.0
    ths_step = 0.1
    ths_stop = 1.0
    ths = np.arange(ths_start,ths_stop,ths_step)
    ## Create main dir
    if mode=="agg":
        pred_path = "predictions_agg"
    elif mode=="loo":
        pred_path = "predictions_loo"
    elif mode=="mask":
        pred_path = "predictions_masked/manual"  
        ths = [0.5]
        # ths=[0.0,0.0000125,0.000025,0.00005,0.000075,0.0001,0.00015,0.0002,0.00025,0.0003,0.0004,0.008]
    elif mode=="regular":
        pred_path="predictions"
        ths=[manual_th]
    if compound is not None:
        pred_path = "{}_{}".format(pred_path,compound)
    
    if results_save_path is None:    
        dir_path = "{}/{}".format(model_path,pred_path)
    else:
        dir_path = results_save_path
        
    create_dir_if_not_exist(dir_path)
    
    # results of pcc for the prediction from original and noisy input by th
    pcc_results = DatasetMetadataSCV("{}/pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(ths)
    
    # results of the size of the mask by th
    mask_results = DatasetMetadataSCV("{}/mask_size_resuls.csv".format(dir_path))
    mask_results.create_header(ths)
    
    #the % of the intersection of true organelle pixels and the mask pixels by th
    context_results = DatasetMetadataSCV("{}/context_resuls.csv".format(dir_path))
    context_results.create_header(ths)
    if save_histo:
        histos = DatasetMetadataSCV("{}/histograms.csv".format(dir_path))
        bins = np.linspace(0,1.0,num_bins)
        histos.create_header(bins)
        histos_l = int(np.ceil(np.sqrt(len(images)+1)))
        histos_fig, histos_axs = plt.subplots(histos_l, histos_l, sharey=True)
        histos_fig.set_figheight(40)
        histos_fig.set_figwidth(40)
        sum_histos = np.zeros_like(bins)
    count=0
    if len(images) > len(dataset.df.data):
        images = range(len(dataset.df.data))
    for image_index in images:
        print("image index: {}/{}".format(count+1,len(images)))
        pccs = []
        mask_sizes = []
        context=[]
        ## Create image dir
        if save_image == True or image_index<save_image:
            create_dir_if_not_exist("{}/{}".format(dir_path,image_index))
        
        ## Preprocess images
        px_start=center_xy[0]-margin[0]-xy_step
        py_start=center_xy[1]-margin[1]-xy_step
        pz_start = 0
        px_end=center_xy[0]+margin[0]+xy_step
        py_end=center_xy[1]+margin[1]+xy_step
        
        # slice_by = None
        slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        if slice_by is None:
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
            # mem_seg_image = np.ones_like(target_image)
            mem_seg_image = mem_seg_image/255.
        else:
            mem_seg_image = np.ones_like(target_image)
        
        weights = None
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
        if weighted_pcc:
            target_seg_image_dilated = np.copy(target_seg_image)
            for h in range(target_seg_image.shape[1]):
                target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((25,25)))  
        else:
            target_seg_image_dilated = None
            
        ## Batch predict
        ## Predicte unet and mask
        print("batch predict...")
        unet_patchs_p = predict(model.unet,input_patchs,batch_size,gv.patch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        mask_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)#model.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        
        ## Create noise vector
        normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float64)
        # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
        
        ## Back to image 
        weights = get_weights(input_patchs[0].shape)
        unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        # mem_seg_image = np.ones_like(unet_p)
        for th in ths:
            try:
                print(th)
                print("mask predict ...")
                if save_image == True or image_index<save_image:
                    if type(th) == str:
                        base_save = "{}/{}/{}/".format(dir_path,image_index,th)
                    else:
                        base_save = "{}/{}/{}/".format(dir_path,image_index,'{:.2f}'.format(float(th)))
                    create_dir_if_not_exist(base_save)
                if mode=="mask" and mask_image is not None:
                    # mask_image_ndarray = target_seg_image
                    if type(mask_image) == str:
                        mask_image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(mask_image))/255.
                    else:
                        mask_image_ndarray = mask_image/255.
                        
                    ## mask_p = (1. - mask_image_ndarray*0.0)
                    mask_p = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.0),tf.float64).numpy()
                    mask_p_binary = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.0),tf.float64).numpy()
                elif mode=="agg":
                    mask_p_binary = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                    mask_p = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                elif mode=="loo":
                    ## 1.0 no noise, 0.0 is noise
                    mask_p_binary = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.0),tf.float64).numpy()
                    mask_p = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.0),tf.float64).numpy()
                else:
                    if manual_th != "full":
                        mask_p_binary = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                        mask_p = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                    else:
                        mask_p_binary = np.ones_like(mask_p_full)
                        mask_p = mask_p_full/d
                mask_size = np.sum(mask_p_binary,dtype=np.float64)
                mask_organelle_intersection = np.sum(mask_p_binary*target_seg_image,dtype=np.float64)/mask_size
                mask_size = mask_size/np.prod(mask_p_binary.shape)
                
                ## Create noisy input and predict unet
                mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p, gv.patch_size, xy_step, z_step)
                mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
                input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
                unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size,gv.patch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs_p),batch_size=batch_size)
        
                ## Back to image 
                input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
                
                ## Save images
                if save_image == True or image_index<save_image:
                    print("saving mask images...")
                    ImageUtils.imsave((mask_p).astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index))  
                # pcc = pearson_corr((unet_p*mem_seg_image/d)[:,:,:], (unet_noise_p*mem_seg_image/d)[:,:,:])
                pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:],target_seg_image_dilated)
                pcc_gt = pearson_corr((unet_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
                pcc_gt_noise = pearson_corr((unet_noise_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
                pccs.append(pcc)
                mask_sizes.append(mask_size)
                c = 1/(mask_organelle_intersection/mask_size)
                context.append(c)
                print("pearson corr for image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc,mask_size,c))
                print("pearson corr for gt image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc_gt,mask_size,c))
                print("pearson corr for noise gt image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc_gt_noise,mask_size,c))
                        
                del unet_noise_patchs_p
                del input_p
                del unet_noise_p
                del mask_noise_patchs
                del mask_patchs_p_term
            except Exception as e:
                print(e)
                pccs.append(-1)
                mask_sizes.append(-1)
                context.append(-1)
            
        if save_image == True or image_index<save_image:
            print("saving global images...")
            image_save = "{}/{}/".format(dir_path,image_index)
            ImageUtils.imsave((input_image).astype(np.float16),"{}/input_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave((target_image).astype(np.float16),"{}/target_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(nuc_image.astype(np.float16),"{}/nuc_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(mem_image.astype(np.float16),"{}/mem_{}.tiff".format(image_save,image_index))  
            ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(image_save,image_index))   
            ImageUtils.imsave((target_seg_image).astype(np.float16),"{}/seg_target_{}.tiff".format(image_save,image_index))
        pcc_results.add_row(pccs)
        mask_results.add_row(mask_sizes)
        context_results.add_row(context)
        if save_histo:
            h, _ = np.histogram((mask_p).reshape(1,-1), bins=num_bins, density=True) #ignore 0.0 for single cells
            sum_histos += h
            histos.add_row(h)
            histos_axs[count // histos_l, count % histos_l].plot(bins,h)
            histos_axs[count // histos_l, count % histos_l].set_title(image_index)
            histos.create()
            histos_fig.savefig("{}/histograms.png".format(dir_path))
        count+=1  
    if save_results: 
        pcc_results.create()
        mask_results.create()
        context_results.create()
    if save_histo:
        avg_histo = sum_histos/len(images)
        histos_axs[(count) // histos_l, (count) % histos_l].plot(bins,avg_histo)
        histos_axs[(count) // histos_l, (count) % histos_l].set_title("average")    
        histos.add_row(avg_histo)
        histos.create()
        histos_fig.savefig("{}/histograms.png".format(dir_path))
    
    return pcc_results, mask_results

# analyze_th("regular",save_histo=False,save_image=1,weighted_pcc=weighted_pcc)
# analyze_th(mode="agg", mask_image=None, manual_th="full",save_image=5,save_histo=False,weighted_pcc=weighted_pcc)
# for mth in [0.85]:
    # analyze_th("regular",mask_image=None,manual_th=mth,save_image=True,save_histo=False)
# analyze_th("mask","{}/X_gradcam_layer_downsample_4_full.tiff".format(gv.model_path)) #gbp_full.tiff #saliency_full.tiff #X_gradcam_layer_downsample_4_full.tiff
# analyze_th("mask","{}/predictions_agg/1/MASK_4.tif".format(gv.model_path))
# analyze_th("mask","{}/predictions_masked/4/MASK_2.tif".format(gv.model_path))
# analyze_th("agg")
# analyze_th("loo")
