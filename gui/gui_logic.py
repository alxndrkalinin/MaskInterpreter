import gc
import tensorflow as tf
from scipy import signal
import tensorflow.keras as keras
from dataset import DataGen
from utils.metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from utils.metrics import *
from dataset import DataGen
import global_vars as gv
from utils.utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from figures.figure_config import figure_config

root_dir = gv.CWD

xy_step = 64
z_step = 16

def get_roi(roi_mode, roi_args, signal):
    if roi_args == None:
        roi_args = [0]
    if roi_mode == "full":
        roi = FullRoi(*roi_args, signal)
    elif roi_mode == "pixel":
        roi = PixelRoi(*roi_args, signal)
    elif roi_mode == "subset":
        roi = SubsetRoi(*roi_args, signal)
    return roi

class PixelRoi():
    def __init__(self, i, j, k, signal):
        self.roi = np.zeros(signal.shape)
        self.roi[:, j, i] = 1

class FullRoi():
    def __init__(self, k, signal):
        self.roi = np.ones_like(signal)

class SubsetRoi():
    def __init__(self, i, j, i2, j2, k, signal):
        self.roi = np.zeros(signal.shape)
        self.roi[:, j:j2, i:i2] = 1

def add_cam_image(img, mask, i):
    alpha = 0.15
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = np.float32(np.dstack([img, img, img]))
    cv2.addWeighted(heatmap, alpha, cam, 1 - alpha, 0, cam)
    cam = cam / np.max(cam)
    ImageUtils.imsave(np.uint8(255 * cam),
                      "{}/{}_{}.tiff".format(root_dir, "test", i))

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
    return tf.nn.relu(x), grad

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = self.modify_model(model)

    def modify_model(self,model):
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer,'activation')]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu or layer.activation == tf.nn.relu:
                layer.activation = guidedRelu
        return model
    
    def __call__(self, input, roi):
        if np.sum(roi)>0:
            input_tensor = tf.Variable(np.expand_dims(input,axis=0), dtype=tf.float16)
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                pred = self.model(input_tensor, training=False)*roi
                loss = tf.reduce_sum(pred)
                            
            grads = tape.gradient(loss, input_tensor)
            dgrad_abs = tf.math.abs(grads)


            ## normalize to range between 0 and 1
            arr_min, arr_max  = np.min(dgrad_abs), np.max(dgrad_abs)
            grad_eval = (dgrad_abs - arr_min) / (arr_max - arr_min + 1e-18)
            return grad_eval.numpy()[0]
        else:
            return roi

class Saliency:
    def __init__(self,model):
        self.model = model
    
    def __call__(self, input, roi):
        if np.sum(roi)>0:
            input_tensor = tf.Variable(np.expand_dims(input,axis=0), dtype=tf.float16)
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                pred = self.model(input_tensor, training=False)*roi
                loss = tf.reduce_sum(pred)
                            
            grads = tape.gradient(loss, input_tensor)
            dgrad_abs = tf.math.abs(grads)


            ## normalize to range between 0 and 1
            arr_min, arr_max  = np.min(dgrad_abs), np.max(dgrad_abs)
            grad_eval = (dgrad_abs - arr_min) / (arr_max - arr_min + 1e-18)
            return grad_eval.numpy()[0]
        else:
            return roi

class MaskInterperter:
    def __init__(self,model):
        self.model = model
    
    def __call__(self, input, roi):
        if np.sum(roi)>0:
            input_tensor = tf.Variable(np.expand_dims(input,axis=0), dtype=tf.float32)
            mask = self.model(input_tensor)*roi
            return mask[0].numpy()
        else:
            return roi
        
class GradCam:
    def __init__(self,model,target_layer,X_gradcam=False):
        self.model = self.modify_model(model,target_layer)
        self.X_gradcam = X_gradcam

    def modify_model(self,model,target_layer):
        return tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    def __call__(self, input, roi):
        if np.sum(roi)>0:
            input_tensor = tf.Variable(np.expand_dims(input,axis=0), dtype=tf.float16)
            with tf.GradientTape() as tape:
                layer_output, pred = self.model(input_tensor)
                loss = tf.reduce_sum(pred*roi)
                
            grads = tape.gradient(loss, layer_output)
            
            if self.X_gradcam:
                pooled_grads = tf.reduce_sum(grads*layer_output, axis=(0, 1, 2, 3))/tf.reduce_sum(layer_output, axis=(0, 1, 2, 3))
            else:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
                
            layer_output = layer_output[0]
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize the heatmap
            heatmap = heatmap.numpy().astype(np.float32)
            heatmap = zoom(heatmap, np.array(input.shape[:-1])/np.array(heatmap.shape))
            return np.expand_dims(heatmap,axis=-1)
        else:
            return roi

def load_model(model_path):
    ## Load model
    print("Loading model:",model_path)
    model = keras.models.load_model(model_path)
    return model

def get_dataset(ds_path):
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    return dataset

def item_from_dataset(dataset,image_index,slice_by=None):
    ## input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image
    input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image =  preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
    if target_seg_image is not None:
        target_seg_image = target_seg_image/255.
    else:
        target_seg_image = np.zeros_like(target_image)
        
    if mem_seg_image is not None:
        # mem_seg_image = np.ones_like(target_image)
        mem_seg_image = mem_seg_image/255.
    else:
        mem_seg_image = np.ones_like(target_image)
    return input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image
    
def _get_weights(shape):
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        values = signal.triang(size)
        weights = weights * values[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)

def get_noise_prediction(signal,mask,mask_th,unet_model,roi_mode="full",roi_args=None, noise_scale = 1.5):
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    px_start = 0
    py_start = 0
    pz_start = 0
    px_end = signal.shape[1]
    py_end = signal.shape[2]
    pz_end = signal.shape[0]
    
    roi = None
    
    roi = get_roi(roi_mode, roi_args, signal)
    mask_after_th = np.where(mask>mask_th, 1.0, 0.0)
    input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,signal,gv.patch_size,xy_step,z_step)
    roi_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,roi.roi,gv.patch_size,xy_step,z_step)
    mask_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_after_th,gv.patch_size,xy_step,z_step)
    noise_patchs = tf.random.normal(tf.shape(mask_patchs),stddev=noise_scale,dtype=tf.float16)
    mask_noise_patchs = (noise_patchs*(1-mask_patchs))
    input_noise_patchs = (mask_patchs*input_patchs)+mask_noise_patchs    
    

    del mask_patchs
    del noise_patchs
    del mask_noise_patchs
    
    noise_prediction_patchs = unet_model.predict(input_noise_patchs,batch_size=4)*roi_patchs

    weights = _get_weights(noise_prediction_patchs[0].shape)
    noise_prediction , d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[noise_prediction_patchs,np.ones_like(input_patchs)],weights,signal.shape,gv.patch_size,xy_step,z_step)
    
    del input_patchs
    del roi_patchs
    del input_noise_patchs
    del noise_prediction_patchs
    del weights
    
    noise_prediction = np.nan_to_num(noise_prediction/d) #/d
    noise_prediction = ImageUtils.normalize(noise_prediction + 0.0001, 1.0, np.float32)    
    return noise_prediction
    
def get_mask(interpreter,signal,roi_mode="full",roi_args=None):
    
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    px_start = 0
    py_start = 0
    pz_start = 0
    px_end = signal.shape[1]
    py_end = signal.shape[2]
    pz_end = signal.shape[0]
    roi = None
    
    roi = get_roi(roi_mode, roi_args, signal)

    input_norm = ImageUtils.normalize(signal, 1.0, np.float32)  ## for saving
    input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,signal,gv.patch_size,xy_step,z_step)
    roi_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,roi.roi,gv.patch_size,xy_step,z_step)
    mask_patchs = []
    for i in range(input_patchs.shape[0]):
        mask_patchs.append(interpreter(input_patchs[i],roi_patchs[i]))
    mask_patchs = np.array(mask_patchs)
    weights = _get_weights(input_patchs[0].shape)
    mask , d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[mask_patchs,np.ones_like(input_patchs)],weights,signal.shape,gv.patch_size,xy_step,z_step)
    
    del input_patchs
    del mask_patchs
    del weights
    
    mask_norm = np.nan_to_num(mask/d) #/d
    mask_norm = ImageUtils.normalize(mask_norm + 0.0001, 1.0, np.float32)
    
    ## increase signal for some methods
    if isinstance(interpreter,Saliency) or isinstance(interpreter,GuidedBackprop):
        mask_norm = mask_norm*3
        
    mask_norm = np.minimum(mask_norm,1.0)
    alpha = 0.25
    full_cam = np.zeros(shape=(*input_norm.shape[:-1], 3))
    for j in range(input_norm.shape[0]):
        img = input_norm[j, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm[j, :, :]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = np.float32(np.dstack([img, img, img]))
        cv2.addWeighted(heatmap, alpha, cam, 1 - alpha, 0, cam)
        full_cam[j, :, :] = cam
    mask = np.uint8(255 * full_cam)
    return mask, mask_norm

def save_mask(root_dir,mask, layer_name, roi_mode, roi_args, method):
    mask = np.float16(mask)
    roi = None
    if roi_mode == "full":
        roi = roi_mode
    else:
        roi = "{}_{}".format(roi_mode,str(roi_args))
    if method == "gradcam" or method=="X_gradcam":
        ImageUtils.imsave(
            mask, "{}/{}_layer_{}_{}.tiff".format(root_dir, method, layer_name, roi))
    else:
        ImageUtils.imsave(
            mask, "{}/{}_{}.tiff".format(root_dir, method, roi))

def predict(model, signal):
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    px_start = 0
    py_start = 0
    pz_start = 0
    px_end = signal.shape[1]
    py_end = signal.shape[2]
    pz_end = signal.shape[0]
    input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,signal,gv.patch_size,xy_step,z_step)
    predict_patchs = model.predict(input_patchs,batch_size=4)
    weights = _get_weights(input_patchs[0].shape)
    prediction , d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[predict_patchs,np.ones_like(predict_patchs)],weights,signal.shape,gv.patch_size,xy_step,z_step)
    
    del input_patchs
    del predict_patchs
    del weights
    
    return prediction/d

def collect_layers(model):

    layers = {}
    for layer in model.layers:
        if "conv" in layer.name:
            layers[layer.name] = layer
    return layers

def get_pearson_per_slice(pred,target,weights=None):
    scores = []
   
    for i in range(pred.shape[0]):
        if weights is not None:
            scores.append(pearson_corr(pred[i:i+1,:],target[i:i+1,:],weights[i:i+1,:]))
        else:
            scores.append(pearson_corr(pred[i:i+1,:],target[i:i+1,:]))
                      
    return scores

def plot_pearson_per_slice(scores, organelle):
    ax = plt.subplot(1, 1, 1)
    x = np.arange(len(scores))
    y1 = scores
    ax.plot(x,y1)
    ax.set_ylabel('Slice Pearson score')
    ax.set_xlabel('Slice')
    ax.set_title("{} Pearson score per slice".format(organelle))
    plt.figtext(0.9, 0.5, "Pearson score of image:{}".format(np.mean(scores)*len(scores)), fontsize=8)
    plt.legend(["Pearson score of slice"])
    plt.show()

def plot_pearson_graph(scores, organelle):
    ax = plt.subplot(1, 1, 1)
    x = np.arange(len(scores))
    y = scores
    m = np.ones(len(scores))*np.mean(scores)
    sp = m + np.ones(len(scores))*np.std(scores)
    sm  = m - np.ones(len(scores))*np.std(scores)
    ax.scatter(x,y)
    ax.plot(x,m,color='blue', linestyle='dashed',label="mean")
    ax.plot(x,sp,color='orange', linestyle='dashed',label="std")
    ax.plot(x,sm,color='orange', linestyle='dashed')
    ax.set_ylabel('Pearson score')
    ax.set_xlabel('Image number')
    ax.set_title("{} Pearson scores".format(organelle))
    plt.figtext(0.9, 0.5, "mean:{}\nstd:{}".format(np.mean(scores),np.std(scores)), fontsize=8)
    plt.legend()
    plt.show()
    
def plot_evaluation_graph(fname,noise_levels, scores, interperters_names):
    ax = plt.subplot(1, 1, 1)
    cmap = plt.cm.get_cmap("tab10")
    colors = cmap.colors  # type: list
    ax.set_prop_cycle(color=colors)
    
    x = noise_levels*100
    for i in range(len(interperters_names)):
        y = scores[i,:]
        ax.plot(x,y,label=interperters_names[i])
    ax.set_ylabel('PCC',fontsize=figure_config["axis"],fontname=figure_config["font"])
    ax.set_xlabel('Noise percentage',fontsize=figure_config["axis"],fontname=figure_config["font"])
    ax.set_title("Pearson scores of different methods with different noise levels")
    plt.savefig(fname)
    plt.legend()
    plt.show()
    
def plot_evaluation_graph_std(fname,interpreters_names,legend=True,ylabel=True):
    data = pd.DataFrame({})
    for i in range(len(interpreters_names)):
        scores_long = load_scores(interpreters_names[i])
        data = pd.concat([data,scores_long],axis=0)
    # plt.rcParams['xtick.labelsize'] = 24     # X-axis tick label font size
    # plt.rcParams['ytick.labelsize'] = 24     # Y-axis tick label font size
    graph = sns.relplot(data=data, kind="line",x="Noise percentage", y="Pearson",hue="Interpreter",legend=legend, aspect=1.0, height=8)
    for ax in graph.axes.flat:
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        if ylabel == False:
            ax.set_xlabel('')                # Remove y-axis label text
            ax.xaxis.set_visible(False)      # Hide the entire y-axis (ticks, labels)
    plt.xlabel("Noise percentage", fontsize=24,fontname=figure_config["font"])
    plt.ylabel("PCC", fontsize=24,fontname=figure_config["font"])
    graph.fig.subplots_adjust(left=0.20, bottom=0.2) 

    graph.figure.savefig(fname)
   
def evaluate_interperters(dir_path,dataset,unet_model,mg_model,selected_layer,X_gradcam, noise_scale):
    
    create_dir_if_not_exist("{}/compare".format(dir_path))
    
    interperters = {
        "mask_interperter":MaskInterperter(model=mg_model),
        "saliency": Saliency(unet_model),
        "gradcam":GradCam(model=unet_model,target_layer=selected_layer,X_gradcam=X_gradcam),
        
    }
    
    noise_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # noise_levels = [0.6,0.7]
    
    mean_pcc_results = DatasetMetadataSCV("{}/mean_evaluation_results_pearson.csv".format(dir_path))
    mean_w_pcc_results = DatasetMetadataSCV("{}/mean_evaluation_results_w_pearson.csv".format(dir_path))
    mean_pcc_results.create_header(noise_levels)
    mean_w_pcc_results.create_header(noise_levels)      
    
    for key in interperters.keys():
        print(key)
        pcc_results = DatasetMetadataSCV("{}/{}_evaluation_results_pearson.csv".format(dir_path,key))
        w_pcc_results = DatasetMetadataSCV("{}/{}_evaluation_results_w_pearson.csv".format(dir_path,key))
        pcc_results.create_header(noise_levels)
        w_pcc_results.create_header(noise_levels)  
              
        interpreter = interperters[key]
        create_dir_if_not_exist("{}/compare/{}".format(dir_path,key))
        for i in range(min(10,dataset.df.data.shape[0])):
            base_dir = "{}/compare/{}/{}".format(dir_path,key,i)
            create_dir_if_not_exist(base_dir)
            print(i)
            pcc = []
            w_pcc = []
            signal, target, target_seg_image, dna, membrane, mem_seg_image = item_from_dataset(dataset, i)
            target_seg_image_dilated = np.copy(target_seg_image)
            for h in range(target_seg_image.shape[1]):
                target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((17,17)))           
            prediction = predict(unet_model,signal)
            prediction_mm =  ImageUtils.normalize(prediction,1.0,np.float32)

            _, mask_norm = get_mask(interpreter,signal)
            ImageUtils.imsave((mask_norm).astype(np.float16),"{}/mask_norm_{}.tiff".format(base_dir,i))
            ImageUtils.imsave((signal).astype(np.float16),"{}/input_{}.tiff".format(base_dir,i))
            ImageUtils.imsave((prediction_mm).astype(np.float16),"{}/prediction_{}.tiff".format(base_dir,i))
            ImageUtils.imsave((target).astype(np.float16),"{}/target_{}.tiff".format(base_dir,i))
            ImageUtils.imsave((target_seg_image).astype(np.float16),"{}/target_seg_{}.tiff".format(base_dir,i))
            for noise_level in noise_levels:
                print("{}-{}".format(i, noise_level))
                mask_th = 0.0
                upper_th = 1.0
                lower_th = 0.0
                mask_noise_level = 0.0
                count = 0
                while not (mask_noise_level < noise_level*1.05 and mask_noise_level > noise_level*0.95) and count<100:
                    count+=1
                    real_mask_th = mask_th
                    mask_noise_level = 1 - np.mean(np.where(mask_norm>mask_th, 1.0, 0.0))
                    if mask_noise_level < noise_level:
                        lower_th = mask_th
                        mask_th = mask_th + ((upper_th-lower_th)/2)
                    else:
                        upper_th = mask_th
                        mask_th = mask_th - ((upper_th-lower_th)/2)
                         
                noise_prediction = get_noise_prediction(signal,mask_norm,real_mask_th,unet_model, noise_scale = noise_scale)
                score = pearson_corr(prediction_mm, noise_prediction)
                pcc.append(score)
                weight_score = pearson_corr(prediction_mm, noise_prediction,target_seg_image_dilated)
                w_pcc.append(weight_score)
                ImageUtils.imsave((np.where(mask_norm>real_mask_th, 1.0, 0.0)).astype(np.float16),"{}/mask_{}_noisevol_{}.tiff".format(base_dir,i,noise_level))
                ImageUtils.imsave((noise_prediction).astype(np.float16),"{}/noisy_prediction_{}_noisevol_{}.tiff".format(base_dir,i,noise_level))
            pcc_results.add_row(pcc)
            w_pcc_results.add_row(w_pcc)
            
        mean_pcc = pcc_results.data.mean(axis=0)
        mean_w_pcc = w_pcc_results.data.mean(axis=0)
    
        mean_pcc_results.add_row(mean_pcc)
        mean_w_pcc_results.add_row(mean_w_pcc)
    
        pcc_results.create()
        w_pcc_results.create()
    
    mean_pcc_results.create()
    mean_w_pcc_results.create()
    
    plot_evaluation_graph("{}/mean_evaluation_results_pearson.svg".format(dir_path),noise_levels, mean_pcc_results.data.values, list(interperters.keys()))
    plot_evaluation_graph("{}/mean_evaluation_results_w_pearson.svg".format(dir_path),noise_levels, mean_w_pcc_results.data.values, list(interperters.keys()))

def load_scores(interpreter):
    scores = pd.read_csv("{}/{}_evaluation_results_pearson.csv".format(gv.model_path,interpreter))
    scores.insert(0,"Interpreter",interpreter)
    scores = scores.drop('Unnamed: 0',axis=1)
    scores["id"] = scores.index
    # Function to rename columns
    def rename_columns(column_name):
        if "0." in column_name:
            return '**'+str(int(float(column_name) * 100))
        return column_name

    # Apply the renaming function to the columns
    scores.columns = [rename_columns(col) for col in scores.columns]
    scores_long = pd.wide_to_long(scores, ["**"], i="id", j="Noise percentage")
    scores_long = scores_long.rename(columns={'**': 'Pearson'})
    return scores_long

# gv.mg_model_path = os.path.join(gv.CWD, "mg_model_mito_10_06_22_5_0_new")
# plot_evaluation_graph_std("{}/comparison.svg".format(gv.mg_model_path),["saliency","gradcam","mask_interperter"])