import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import DataGen
from mg_analyzer import analyze_th
from utils.utils import create_dir_if_not_exist
import global_vars as gv
from cell_imaging_utils.image.image_utils import ImageUtils
import os
from figure_config import figure_config, scalebar


params = [
    {"organelle": "Nucleolus-(Granular-Component)", "model": "../mg_model_ngc_13_05_24_1.5", "noise": 1.5, "th": 0.6},
    {"organelle": "Endoplasmic-reticulum", "model": "../mg_model_er_13_05_24_1.5", "noise": 1.5, "th": 0.4},
    {"organelle": "Golgi", "model": "../mg_model_golgi_13_05_24_1.5", "noise": 1.5, "th": 0.3},
    {"organelle": "Actomyosin-bundles", "model": "../mg_model_bundles_13_05_24_1.0", "noise": 1.0, "th": 0.02},
    {"organelle": "Mitochondria", "model": "../mg_model_mito_13_05_24_1.5", "noise": 1.5, "th": 0.2},
    {"organelle": "Nuclear-envelope", "model": "../mg_model_ne_13_05_24_1.0", "noise": 1.0, "th": 0.2},
    {"organelle": "Microtubules", "model": "../mg_model_microtubules_13_05_24_1.5", "noise": 1.5, "th": 0.1},
    {"organelle": "Actin-filaments", "model": "../mg_model_actin_13_05_24_1.5", "noise": 1.5, "th": 0.2},
    {"organelle": "Plasma-membrane", "model": "../mg_model_membrane_13_05_24_1.5", "noise": 1.5, "th": 0.4},
    {"organelle": "DNA", "model": "../mg_model_dna_13_05_24_1.5b", "noise": 1.5, "th": 0.3},
]

slice_num = 42 #for Endosomes_4 slice 23, for NE_0 slice 28

def auto_balance(image):
    image = image.astype(np.float32)
    plow, phigh = np.percentile(image, (0.1, 99.9))
    image = np.clip((image - plow) / (phigh - plow), 0, 1)
    return image

def collect_images(save_dir, param, image_index=0):
    base_path = f"{save_dir}/{param['organelle']}/{image_index}"
    file_paths = {
        "prediction": f"{base_path}/unet_prediction_{image_index}.tiff",
        "input": f"{base_path}/input_{image_index}.tiff",
        "nuc": f"{base_path}/nuc_{image_index}.tiff",
        "mem": f"{base_path}/mem_{image_index}.tiff",
        "mask": f"{base_path}/{param['th']:.2f}/mask_{image_index}.tiff",
        "noisy_prediction": f"{base_path}/{param['th']:.2f}/noisy_unet_prediction_{image_index}.tiff"
    }

    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

    stacks = {key: ImageUtils.normalize(ImageUtils.image_to_ndarray(ImageUtils.imread(path)), max_value=1.0, dtype=np.float32)
              for key, path in file_paths.items()}

    best_slice_index = slice_num
    most_informative_slices = {key: auto_balance(stack[best_slice_index, :, :, 0]) for key, stack in stacks.items()}
    return most_informative_slices["input"],most_informative_slices["nuc"],most_informative_slices["mem"], most_informative_slices["mask"], most_informative_slices["prediction"], most_informative_slices["noisy_prediction"], best_slice_index

def get_mask_efficacy_score(results_path):
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        return results_df.iloc[0].values[1]
    return None

def generate_prediction_image(dataset, params, save_dir):
    create_dir_if_not_exist(save_dir)
    num_rows = 4
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(17, 12))
    fig.suptitle("Predictions from pipeline 4.0", fontsize=figure_config["title"], fontname=figure_config["font"],y=0.95)
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    # Hide the remaining cells in the first row
    for col in range(1, 5):
        axes[0, col].axis('off')

    # Display predictions on the second and third rows
    for i, ax in enumerate(axes.flatten()):
        try:
            param = params[i]
            model_path = param["model"]
            noise_scale = param["noise"]
            threshold = param["th"]
            results_save_path = f"{save_dir}/{param['organelle']}"
            
            analyze_th(dataset, mode="regular", manual_th=threshold, save_image=True, save_histo=False, weighted_pcc=False, 
                       model_path=model_path, images=range(1), noise_scale=noise_scale, save_results=True, results_save_path=results_save_path)

            # Display the input image in the first subplot (1st row, 1st column)
            input_image, nuc_image, mem_image, _, prediction_original, _, _ = collect_images(save_dir, param)  # Assuming first param's path for input image
            if i ==0:
                axes[0, 0].imshow(input_image, cmap='gray')
                axes[0, 0].set_title("Input Image",fontsize=figure_config["organelle"], fontname=figure_config["font"])
                axes[0, 0].axis('off')
                axes[0, 0].add_artist(scalebar)
                ## Membrane
                axes[3, 0].imshow(mem_image, cmap='gray')
                axes[3, 0].set_title("Plasma-membrane-GT",fontsize=figure_config["organelle"], fontname=figure_config["font"])
                axes[3, 0].axis('off')
                ## DNA
                axes[3, 1].imshow(nuc_image, cmap='gray')
                axes[3, 1].set_title("DNA-GT",fontsize=figure_config["organelle"], fontname=figure_config["font"])
                axes[3, 1].axis('off')
            

            # Determine row and column for each prediction
            row = 1 + (i // 5)  # Rows 1 and 2 (since row 0 is for the input image)
            col = i % 5         # Columns 0 through 4

            # Get the mask efficacy score from results.csv
            mask_efficacy_score = get_mask_efficacy_score(f"{results_save_path}/pcc_resuls.csv")
            axes[row, col].imshow(prediction_original, cmap='gray')
            axes[row, col].axis('off')

            # # Add organelle name and mask efficacy score on the image
            # title_text = "Mask Efficacy:{:.2f}".format(mask_efficacy_score)
            name = param['organelle']
            if i ==0:
                name = "Nucleolus\n(Granular-Component)"
            axes[row, col].set_title(name,fontsize=figure_config["organelle"], fontname=figure_config["font"])
            # axes[row, col].text(0.05, 0.95, title_text, ha='left', va='top', transform=axes[row, col].transAxes,
            #                     color='white', fontsize=10)
        except Exception as e:
            ax.axis('off')
            print("data for subplot {} not exist".format(i))
            
    final_fig_path = f"{save_dir}/input_and_predictions_main.png"
    # plt.tight_layout()
    plt.savefig(final_fig_path,bbox_inches='tight',pad_inches=0.05)
    plt.show()

save_dir = os.path.join( gv.CWD, 'figures/predictions/test') #Endosomes_4
ds_path = f"{save_dir}/dataset.csv"
dataset = DataGen(ds_path, "channel_signal", "channel_target", batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0.0, max_precentage=1.0, augment=False)
generate_prediction_image(dataset, params, save_dir)
