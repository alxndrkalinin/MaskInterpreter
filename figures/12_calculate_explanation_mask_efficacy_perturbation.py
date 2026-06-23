#This script is used to choose noise std for each organelle, 
# we noise the entire image with different noise stds and choose the one 
# that reduce PCC below 0.2

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import analyze_th

params = [
          #{"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5},
          #{"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5},
          #{"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5},
          #{"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5},
          #{"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0},
          #{"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5},
          #{"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5},
          {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5},
          {"organelle":"DNA","model":"../mg_model_dna_13_05_24_1.5b","noise":1.5},
          ]
base_dir = os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation')
gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False

# Create a list of organelles from the parameters
organelles = [
              "Actin filaments_Paclitaxol",
              "Actin filaments_Rapamycin",
              "Actin filaments_s-Nitro-Blebbistatin",
              "Actin filaments_Staurosporine",
              "Actomyosin bundles_Brefeldin",
              "Actomyosin bundles_Rapamycin",
              "Actomyosin bundles_s-Nitro-Blebbistatin",
              "Endoplasmic reticulum_Brefeldin",
              "Endoplasmic reticulum_Paclitaxol",
              "Endoplasmic reticulum_Staurosporine",
              "Golgi_Brefeldin",
              "Golgi_Paclitaxol",
              "Golgi_s-Nitro-Blebbistatin",
              "Golgi_Staurosporine",
              "Lysosome_Brefeldin",
              "Lysosome_Paclitaxol",
              "Lysosome_s-Nitro-Blebbistatin",
              "Microtubules_Paclitaxol",
              "Microtubules_Rapamycin" ,
              "Microtubules_Staurosporine",
              "Tight junctions_Brefeldin",
              "Tight junctions_Paclitaxol",
              "Tight junctions_Rapamycin",
              "Tight junctions_s-Nitro-Blebbistatin",
              "Tight junctions_Staurosporine"
              ]
for organelle in organelles:
    ds_path = "{}/train_test_list/{}/image_list_with_metadata__with_efficacy_scores_full.csv".format(base_dir,organelle)
    if not os.path.exists(ds_path):
      images_list1 = "{}/{}/{}.csv".format(base_dir,organelle,organelle)
      images_list2 = "{}/train_test_list/{}/image_list_test_Vehicle.csv".format(base_dir,organelle)
      images_list3 = "{}/train_test_list/{}/image_list_test_{}.csv".format(base_dir,organelle,organelle.split('_')[-1])
      images_df = pd.read_csv(images_list1)
      images_df2 = pd.read_csv(images_list2)
      images_df3 = pd.read_csv(images_list3)

      # Merge df with df1 and df2 based on the appropriate columns
      df_merged1 = images_df.merge(images_df2, left_on='combined_image_storage_path', right_on='path_tiff', how='left')
      df_merged2 = images_df.merge(images_df3, left_on='combined_image_storage_path', right_on='path_tiff', how='left')

      # Fill NaN values in df_merged1 with values from df_merged2
      df_final = df_merged1.combine_first(df_merged2)

      # Drop the 'path_tiff' column from the final dataframe
      df_final.drop(columns=['path_tiff','Unnamed: 0_x','Unnamed: 0_y'], inplace=True)
      df_final = df_final.dropna(subset=['channel_signal','channel_target','channel_dna','structure_seg'])
      
      # ds_path = "{}/train_test_list/{}/image_list_with_metadata_full.csv".format(base_dir,organelle)
      df_final.to_csv(ds_path,index=False)   
    for param in params:
        print("dataset:",organelle,"model:",param["organelle"]," ",param["model"])
        dataset = DataGen(ds_path ,gv.input,gv.target,image_path_col='combined_image_storage_path',batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
        print("# images in dataset:",dataset.df.data.shape[0])
        if param["model"] in dataset.df.data.columns and (not "override" in param.keys() or not param["override"]):
              print("already calculated...")
              continue
        else:
          pcc_results, mask_results = analyze_th(dataset,"regular",mask_image=None,manual_th="full",save_image=False,save_histo=False,weighted_pcc = weighted_pcc, model_path=param["model"],model=None,compound=None,images=range(dataset.df.data.shape[0]),noise_scale=param["noise"],save_results=False)
          df_final = pd.read_csv(ds_path)
          df_final[param["model"]] = pcc_results.data["full"].values
          df_final.to_csv(ds_path,index=False)

