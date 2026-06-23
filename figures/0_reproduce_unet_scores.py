import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import calc_unet_pcc
from figure_config import figure_config

params = [
          {"organelle":"Nuclear-envelope","model":"../unet_model_22_05_22_ne_128"},
          {"organelle":"Golgi","model":"../unet_model_22_05_22_golgi_128"},
          {"organelle":"Microtubules","model":"../unet_model_22_05_22_microtubules_128"},
          {"organelle":"Endoplasmic-reticulum","model":"../unet_model_22_05_22_er_128"},
          {"organelle":"Plasma-membrane","model":"../unet_model_22_05_22_membrane_128"},
          {"organelle":"Actin-filaments","model":"../unet_model_22_05_22_actin_128"},
          {"organelle":"Nucleolus-(Granular-Component)","model":"../unet_model_22_05_22_ngc_128"},
          {"organelle":"Mitochondria","model":"../unet_model_22_05_22_mito_128"},
          {"organelle":"Actomyosin-bundles","model":"../unet_model_22_05_22_bundles_128"},
          {"organelle":"DNA","model":"../unet_model_22_05_22_dna_128"}
          ]

gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False

def plot_unet_scores():
    fig, ax = plt.subplots(figsize=(12,4),gridspec_kw={'wspace':0.2,'hspace':0.4})

    all_pcc_data = []
    labels = []
    medians = []
    
    for param in params:
        try:
            # Load PCC data for each organelle
            csv_path = "{}/pcc_resuls.csv".format(param["model"])
            pcc_data = DatasetMetadataSCV(csv_path, csv_path).data
            all_pcc_data.append(pcc_data['PCC'])
            labels.append(param["organelle"])
            medians.append(np.median(pcc_data['PCC']))
        except Exception as e:
            print(f"Data for {param['organelle']} does not exist: {e}")

    # Sorting the data based on medians
    sorted_indices = np.argsort(medians)[::-1]
    all_pcc_data = [all_pcc_data[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Creating the boxplot
    bplot = ax.boxplot(all_pcc_data, patch_artist=True, labels=labels)

    ax.set_title("Prediction performance across different\nsubcellular structures (higher is better)",fontsize=figure_config["title"], fontname=figure_config["font"])
    # ax.set_xlabel("Organelles",fontsize=figure_config["axis"], fontname=figure_config["font"])
    ax.set_ylabel("PCC",fontsize=figure_config["axis"], fontname=figure_config["font"])
    plt.xticks(rotation=45, ha='right')

    # plt.tight_layout()
    plt.savefig("../figures/unet_prediction_performance_sorted.png",bbox_inches='tight', pad_inches=0.1)
    plt.show()

for param in params:
    print(param["organelle"])
    ds_path = os.path.join(gv.DATA_PATH, "{}/image_list_test.csv".format(param["organelle"]))
    dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0.0, max_precentage=1.0, augment=False)
    calc_unet_pcc(dataset, model_path=param["model"], weighted_pcc=weighted_pcc,images=range(min(10,dataset.df.data.shape[0])))

plot_unet_scores()
