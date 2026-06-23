import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from cell_imaging_utils.image.image_utils import ImageUtils
from mg_analyzer import analyze_th
from dataset import DataGen
import global_vars as gv
import os
import cv2
from scipy import ndimage
from skimage import morphology
from matplotlib.patches import Rectangle, ConnectionPatch
import tensorflow.keras as keras

params = [
    {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","th":0.2,"slice":30,"noise":1.0,"zoom_rect":[150, 100, 200, 200]},  # Example zoom rectangle (x, y, width, height)}
    {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","th":0.40,"slice":31,"noise":1.5,"zoom_rect":[500, 250, 200, 200]},
]

def auto_balance(image):
    """Auto balance the image similar to ImageJ's Auto Contrast function."""
    image = image.astype(np.float32)
    plow, phigh = np.percentile(image, (0.1, 99.9))
    image = np.clip((image - plow) / (phigh - plow), 0, 1)
    return image

def predict_noisy(param, result_mask, model, dataset):
    pcc_results, mask_results = analyze_th(dataset, "mask", mask_image=np.expand_dims(result_mask, axis=-1) * 255, manual_th="full", save_image=True, save_histo=False, weighted_pcc=False, model=model, compound=None, images=[image_index], results_save_path="{}/{}".format(param["model"], mask_mode_pred_path,noise_scale=param["noise"]))
    return pcc_results, mask_results

def process_mask_image(manual_noise, steps=4):
    """Process the mask image and reduce it in specified steps."""
    # Close holes in the mask
    closed_mask = ndimage.binary_closing((manual_noise[:, :, :, 0] / 255).astype(np.uint8), structure=np.ones((1,3,3))).astype(int)
    closed_mask = ndimage.binary_erosion(closed_mask, structure=np.ones((1,7,7))).astype(int)
    # Create stepwise reductions
    step_reductions = []
    for i in range(steps):
        closed_mask = ndimage.binary_erosion(closed_mask, structure=np.ones((1,11,11))).astype(int)
        step_reductions.append(closed_mask)
    
    return step_reductions

def get_manual_noise(param, image_index):
    base_path = "{}/predictions_agg/{}".format(param["model"], image_index)
    mask_path = "{}/{}/mask_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index)
    seg_gt_path = "{}/seg_target_{}.tiff".format(base_path, image_index)
    
    seg_gt_image = ImageUtils.image_to_ndarray(ImageUtils.imread(seg_gt_path))
    mask_image_no_seg_gt = (ImageUtils.image_to_ndarray(ImageUtils.imread(mask_path))*(1.0-seg_gt_image)).astype(bool)
    mask_image_no_seg_gt = morphology.remove_small_objects(mask_image_no_seg_gt,100)
    return mask_image_no_seg_gt*255

def get_3d_mask(param, image_index):
    base_path = "{}/predictions_agg/{}".format(param["model"], image_index)
    mask_path = "{}/{}/mask_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index)
    
    mask_image = (ImageUtils.image_to_ndarray(ImageUtils.imread(mask_path))).astype(np.uint8)
    return mask_image

def collect_images(param, image_index):
    # Construct paths for the required images
    base_path = "{}/{}/{}".format(param["model"], mask_mode_pred_path, image_index)
    file_paths = {
        "prediction": "{}/unet_prediction_{}.tiff".format(base_path, image_index),
        "input": "{}/input_{}.tiff".format(base_path, image_index),
        "mask": "{}/{}/mask_{}.tiff".format(base_path, '{0:.2f}'.format(0.5), image_index),
        "noisy_prediction": "{}/{}/noisy_unet_prediction_{}.tiff".format(base_path, '{0:.2f}'.format(0.5), image_index),
        "ground_truth": "{}/target_{}.tiff".format(base_path, image_index),
    }

    # Check if files exist
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

    # Load and normalize the stacks
    stacks = {}
    for key, path in file_paths.items():
        stack = ImageUtils.image_to_ndarray(ImageUtils.imread(path))
        stack = ImageUtils.normalize(stack, max_value=1.0, dtype=np.float32)
        stacks[key] = stack

    best_slice_index = param["slice"]

    # Extract the most informative slice from each stack using the index found from the prediction stack
    most_informative_slices = {key: auto_balance(stack[best_slice_index, :, :, 0]) for key, stack in stacks.items()}

    return most_informative_slices["input"],most_informative_slices["noisy_prediction"], best_slice_index

def overlay_images(input_image, mask_image, result_image):
    input_image_rgb = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    turquoise = np.array([64, 224, 208]) / 255.0
    light_green = np.array([0, 224, 0]) / 255.0
    
    # Create an overlay image
    overlay = input_image_rgb.copy().astype(np.float32)
    
    # Apply mask color
    mask_indices = mask_image > 0
    overlay[mask_indices] = 0.5 * turquoise * 255 + 0.5 * overlay[mask_indices]

    result_indices = result_image > 0
    overlay[result_indices] = 0.5 * light_green * 255 + 0.5 * overlay[result_indices]

    return overlay.astype(np.uint8)

def plot_manual_validation(image_index, param, save_path, steps, zoom_rect):
    mask_3d = get_3d_mask(param,image_index)
    manual_noise = get_manual_noise(param,image_index)
    step_reductions = process_mask_image(manual_noise, steps)
    
    fig = plt.figure(figsize=(2*steps,4*1))
    gs = gridspec.GridSpec(4, steps, height_ratios=[1, 1, 1, 1])
    gs.update(wspace=0.1, hspace=0.3)
    model = keras.models.load_model(param["model"])
    ds_path = os.path.join(gv.DATA_PATH, "{}/image_list_test.csv".format(param["organelle"]))
    dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0.0, max_precentage=1.0, augment=False)
    for i, step in enumerate(step_reductions):
        print("step:",i)
        result_mask = np.logical_not(step).astype(np.uint8)
        
        pcc_results, mask_results = predict_noisy(param, result_mask, model, dataset)
        print("pcc_result:",pcc_results.data)
        print("mask_results:",mask_results.data)
        input_image, noisy_prediction, slice_index = collect_images(param, image_index)
        
        overlay = overlay_images(input_image, auto_balance(mask_3d[param["slice"], :, :, 0]), step[param["slice"], :, :])
        
        # Row 1: Overlay images
        ax1 = plt.subplot(gs[0, i])
        ax1.imshow(overlay)
        ax1.axis('off')
        # ax1.text(0.01, 0.99, f"Manual noise", transform=ax1.transAxes, fontsize=10, color='white', ha='left', va='top')
        
        # Row 2: Zoomed-in overlay images
        x, y, w, h = zoom_rect
        zoomed_in_overlay = overlay[y:y+h, x:x+w]
        ax2 = plt.subplot(gs[1, i])
        ax2.imshow(zoomed_in_overlay)
        ax2.axis('off')
        ax2.add_patch(Rectangle((0, 0), w, h, edgecolor='red', facecolor='none', linewidth=4))

        # Add the rectangle and connection lines
        rect = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
        ax1.add_patch(rect)

        con = ConnectionPatch(xyA=(x + (w/2), y+h), xyB=(w/2, 0), coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color='red', linewidth=2, arrowstyle="-|>", zorder=101)
        ax2.add_artist(con)
        
        # Row 3: Noisy predictions
        ax3 = plt.subplot(gs[2, i])
        ax3.imshow(noisy_prediction, cmap='gray')
        ax3.axis('off')
        # ax3.text(0.01, 0.99, f"Noisy Prediction", transform=ax3.transAxes, fontsize=10, color='white', ha='left', va='top')
        
        # Row 4: Zoomed-in noisy predictions
        zoomed_in_prediction = noisy_prediction[y:y+h, x:x+w]
        ax4 = plt.subplot(gs[3, i])
        ax4.imshow(zoomed_in_prediction, cmap='gray')
        ax4.axis('off')
        ax4.add_patch(Rectangle((0, 0), w, h, edgecolor='red', facecolor='none', linewidth=4))

        # Add the rectangle and connection lines
        rect_pred = Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
        ax3.add_patch(rect_pred)

        con = ConnectionPatch(xyA=(x + (w/2), y+h), xyB=(w/2, 0), coordsA="data", coordsB="data",
                              axesA=ax3, axesB=ax4, color='red', linewidth=2, arrowstyle="-|>", zorder=102)
        ax4.add_artist(con)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
# Example usage
image_index = 4
mask_mode_pred_path = "predictions_masked/manual"
steps = 3

# Directory to save individual plots
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)
organelle_names = []

# Generate and save individual organelle plots
for param in params:
    print(param["organelle"])
    organelle_names.append(param["organelle"])
    plot_manual_validation(image_index, param=param, save_path=os.path.join(output_dir, 'manual_validation_{}.png'.format(param["organelle"])), steps=steps, zoom_rect=param["zoom_rect"])
