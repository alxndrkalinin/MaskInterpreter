import os
import numpy as np
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
import global_vars as gv
"""This script will create train and test lists in csv file for unpertrubed data, this script need to be run after the download_and_create script"""
## global vars
#use False if this list is only for 1 organelle and then there will be different csv files for each organelle
#and True if you wnt the same files for all organlles
GLOBAL_LIST = False
#The path to where the metadata of the downloaded images
storage_root = gv.DATA_PATH 
#path to save to lists in the end
if GLOBAL_LIST:
  lists_root_dir = os.path.join(gv.CWD, 'cell_generator')
else:
  lists_root_dir = os.path.join(gv.DATA_PATH, 'train_test_list/')
#what organelles to process
organelles={"Nuclear-envelope":[]} # organelles={"Tight junctions":[],"Golgi":[],"Actomyosin bundles":[],"Microtubules":[],"Actin filaments":[],"Endoplasmic reticulum":[],"Lysosome":[]} ## pertrub
#% of images to be in the trainin set, the rest will be in the test set
train_p = 0.0

#the channel number in the stack image data
target_channel = 3
input_channels = 0
dna_channel = 1
membrane_channel = 2
membrane_seg = 5
structure_seg = 6
#int label to mark the organelle can be used for classifier trying to predict the organelle
organelle_label = {"Golgi":0,"Microtubules":1,"Nuclear-envelope":2,"Actin-filaments":3,"Plasma-membrane":4,"Nucleolus-(Dense-Fibrillar-Component)":5,"Mitochondria":6,"Endoplasmic-reticulum":7,"Tight-junctions":8,"Nucleolus-(Granular-Component)":9,"Actomyosin-bundles":10,"Desmosomes":11,"Lysosome":12,"Adherens-junctions":13,"Gap-junctions":14,"Matrix-adhesions":15,"Peroxisomes":16,"Endosomes":17}

## instructions - channels in outputs images
# 0-FOV roi
# 1 dna raw roi
# 2 membrane raw roi
# 3 structure raw roi
# 4 dna seg roi
# 5 membrane seg roi
# 6 structure seg roi
##

def create_datasets():
    if (not os.path.exists(lists_root_dir)):
        os.makedirs(lists_root_dir)
    first = True   
    for organelle in organelles.keys():
        organelle_metadata_dir = "{}{}/{}.csv".format(storage_root,organelle,organelle)
        if not GLOBAL_LIST:
          if (not os.path.exists("{}{}".format(lists_root_dir,organelle))):
            os.makedirs("{}{}".format(lists_root_dir,organelle))
        print("Opening metadata file {}".format(organelle_metadata_dir))
        organelle_metadata  = DatasetMetadataSCV(organelle_metadata_dir,organelle_metadata_dir)
        organelle_metadata.data = organelle_metadata.data.sample(frac=1,random_state=42).reset_index(drop=True) ##shuffle rows
        print("Finish Opening metadata file {}".format(organelle_metadata_dir))
        num_of_rows = organelle_metadata.get_shape()[0]
        for t in [{"name":"image_list_train.csv","rows":range(int(num_of_rows*train_p))},{"name":"image_list_test.csv","rows":range(int(num_of_rows*train_p),int(num_of_rows*train_p)+int(num_of_rows*(1-train_p)))}]:
            if GLOBAL_LIST:
              image_list_path = "{}/{}".format(lists_root_dir,t["name"])
            else:
              image_list_path = "{}{}/{}".format(lists_root_dir,organelle,t["name"])
            
            if (not GLOBAL_LIST or  (GLOBAL_LIST and (not os.path.exists(image_list_path) or first))):
              image_list = DatasetMetadataSCV(image_list_path)                  
            else:
              image_list = DatasetMetadataSCV(image_list_path,image_list_path)
              
            image_list.create_header(['path_tiff','channel_signal','channel_target','channel_dna','channel_membrane','membrane_seg','structure_seg','label'])
              
            for i in t["rows"]:
                image_path=organelle_metadata.get_item(i,'combined_image_storage_path')
                image_list.add_row([image_path,input_channels,target_channel,dna_channel,membrane_channel,membrane_seg,structure_seg,organelle_label[organelle]])
            
            image_list.create()
            print("Created/Updated {}".format(image_list_path))
        first = False
                
                
                
def run():
    create_datasets()
    
run()