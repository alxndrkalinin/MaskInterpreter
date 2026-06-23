#This script is used to choose noise std for each organelle, 
# we noise the entire image with different noise stds and choose the one 
# that reduce PCC below 0.2

import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import analyze_th
from figure_config import figure_config

params = [
          {"organelle":"Nucleolus\n(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5},
          {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5},
          {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5},
          {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5},
          {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5},
          {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5},
          {"organelle":"DNA","model":"../mg_model_dna_13_05_24_1.5b","noise":1.5},
        #   {"organelle":"DNA","model":"../mg_model_dna_13_05_24_1.5","noise":1.5},
          ]

gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False


def plot_th_analysis():
    # Enhanced setup for 9 subplots, each with additional annotations for the first x-value where PCC < 0.87
    fig, axs = plt.subplots(2,5, figsize=(20,8),gridspec_kw={'wspace':0.2,'hspace':0.4})  # 3x3 grid of subplots

    for i, ax in enumerate(axs.flatten()):
        try:
            # load PCC data for each subplot
            pcc_csv_path = "{}/predictions_agg/pcc_resuls.csv".format(params[i]["model"])
            pcc_data = DatasetMetadataSCV(pcc_csv_path,pcc_csv_path)
            mask_size_csv_path = "{}/predictions_agg/mask_size_resuls.csv".format(params[i]["model"])
            mask_size_data = DatasetMetadataSCV(mask_size_csv_path,mask_size_csv_path)
            ths = np.array(list(map(lambda val: float('{:.2f}'.format(float(val))), pcc_data.data.columns[1:])))
            pcc = pcc_data.data.mean(axis=0)[1:]
            mask_size = 1 - mask_size_data.data.mean(axis=0)[1:]  # Calculating mean mask
            
            # Plotting
            ax.plot(ths, pcc, label='Model Prediction Quality')
            ax.axhline(y=0.87, color='r', linestyle='--', label='Threshold: PCC=0.87')
            ax.set_title(params[i]["organelle"],fontsize=figure_config["organelle"], fontname=figure_config["font"])
            ax.set_xlabel('TH',fontsize=figure_config["axis"], fontname=figure_config["font"])
            ax.set_ylabel('PCC',fontsize=figure_config["axis"], fontname=figure_config["font"])
            # Find the first x-value where PCC < 0.2 and annotate
            over_threshold = ths[pcc > 0.87]
            if over_threshold.size > 0:  # Check if there's any value below the threshold
                last_over = over_threshold[-1]
                ax.axvline(x=last_over, color='g', linestyle=':', label=f'Critical TH: {last_over}')
                mean_mask_vol = mask_size[np.argmin(np.abs(ths - last_over))] * 100
                ax.text(last_over + 0.02, 0.9, f'Mean Mask Vol: {mean_mask_vol:.2f}%', transform=ax.transData, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            ax.legend(loc='lower left')
            if i % 3 !=0:
                ax.get_yaxis().set_visible(False)
            if i < 7:
                ax.get_xaxis().set_visible(False)
        except Exception as e:
            ax.axis('off')
            print("data for subplot {} not exist".format(i))

    # Main title and layout adjustments
    plt.suptitle('Effect of TH the mask on Model Prediction from Noisy Input',fontsize=figure_config["title"], fontname=figure_config["font"])
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    plt.savefig("../figures/find_th.png",bbox_inches='tight', pad_inches=0.1)


for param in params:
    print(param["organelle"])
    ds_path = os.path.join(gv.DATA_PATH, "{}/image_list_test.csv".format(param["organelle"]))
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    print("# images in dataset:",dataset.df.data.shape[0])
    analyze_th(dataset,"agg",mask_image=None,manual_th="full",save_image=5,save_histo=False,weighted_pcc = weighted_pcc, model_path=param["model"],model=None,compound=None,images=range(min(10,dataset.df.data.shape[0])),noise_scale=param["noise"])
plot_th_analysis()


