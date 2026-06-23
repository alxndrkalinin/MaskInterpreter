import os
import threading
import numpy as np
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from cell_imaging_utils.image.image_utils import ImageUtils
import shutil
from collections import OrderedDict
from seg_proto import *
import pandas as pd
import imageio
import global_vars as gv
"""This script will segment and stack the field of views (FOVs) pertrubed data from the Allen Inst. and will segment target organelles,
stack all the relevant channels for that FOV.
The data need to be dowloaded manually from https://www.allencell.org/data-downloading.html#sectionDrugSignatureData"""

## global vars
#number of threads to run
num_threads=1
#where to save processed data
storage_root = os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation')
#temp location to save data that is being processed (SSD memory)
temp_storage_root = os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation/raw')
#path to metadata.csv
datasets_metadata_dir = "{}/drug_perturbation_dataset_2.csv".format(temp_storage_root)
#what organelles to process
organelles={"Golgi":[],"Tight junctions":[],"Actomyosin bundles":[],"Microtubules":[],"Actin filaments":[],"Endoplasmic reticulum":[],"Lysosome":[]}

## instructions - channels in outputs images
# 0-BF roi
# 1 dna raw roi
# 2 membrane raw roi
# 3 structure raw roi
# 4 dna seg roi
# 5 membrane seg roi
# 6 structure seg roi
##

## roi is [z_min,z_max,y_min,y_max,x_min,x_max]
## fov to crop is 1:3 ratio 
## metadata.data.loc[metadata.data['structure_name'] == 'GJA1']
## not found 'ATP2A2','HIST1H2BJ','NUP153','SMC1A','SON'
only_csvs = True
override = True

organelle_segmentor_dict = {'Actin filaments':SegActinFilaments(), 'Lysosome':SegLysosome(), 'Actomyosin bundles':SegMyosin(),
        'Endoplasmic reticulum':SegER(), 'Golgi':SegGolgi(), 'Tight junctions':SegTightJunctions(), 'Microtubules':SegMicrotubules()}

def imread(path):
    image = imageio.imread(path)
    image = np.array(image).astype(np.float32)
    return image

def segment_and_create_image(organelle,fov_path,fov_channel,structure_fl_channel,dna_fl_channel,mem_fl_channel,new_image_path):
## instructions - channels in outputs images
# 0-FOV roi
# 1 dna raw roi
# 2 membrane raw roi
# 3 structure raw roi
# 4 dna seg roi
# 5 membrane seg roi
# 6 structure seg roi
##
    if (not only_csvs):
        if ((not override) and os.path.exists(new_image_path)):
            print("image already exists: {}".format(new_image_path))
            return
            
        print("Creating image {}".format(new_image_path))
        new_image = None
          
        fov_ndarray = imread(fov_path)
        if len(fov_ndarray.shape) == 5:
            fov_ndarray = fov_ndarray[0]
        if len(fov_ndarray.shape) == 6:
            fov_ndarray = fov_ndarray[0,0]
        sliced_fov_ndarray = ImageUtils.get_channel(fov_ndarray,int(fov_channel))
        new_image = sliced_fov_ndarray     
        
        dna_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(dna_fl_channel))
        new_image = ImageUtils.add_channel(new_image,dna_channel_ndarray)
        
        mem_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(mem_fl_channel))
        new_image = ImageUtils.add_channel(new_image,mem_channel_ndarray)
        
        structure_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(structure_fl_channel))
        new_image = ImageUtils.add_channel(new_image,structure_channel_ndarray)
           
        dna_seg_ndarray = np.zeros_like(dna_channel_ndarray)
        new_image = ImageUtils.add_channel(new_image,dna_seg_ndarray)
             
        mem_seg_ndarray = np.zeros_like(mem_channel_ndarray)
        new_image = ImageUtils.add_channel(new_image,mem_seg_ndarray)
        
        
        structure_seg_ndarray = np.expand_dims(organelle_segmentor_dict[organelle].segment(structure_channel_ndarray[0]),axis=0)
        new_image = ImageUtils.add_channel(new_image,structure_seg_ndarray)
        
         
        print("saving image: {}".format(new_image_path))
        imageio.imwrite(new_image_path,new_image.astype(np.float16))
        print("saved image: {}".format(new_image_path))

def cretae_metadata_file(metadata,organelle,drug):
    organelle_dir = "{}/{}_{}".format(storage_root,organelle,drug)
    organelle_metadata_path = "{}/{}_{}.csv".format(organelle_dir,organelle,drug)
    print("Creating {}".format(organelle_metadata_path))
    if (not os.path.exists(organelle_dir)):
        os.makedirs(organelle_dir)
    if (os.path.exists(organelle_metadata_path)):
        os.remove(organelle_metadata_path)
        
    organelle_metadata = DatasetMetadataSCV(organelle_metadata_path)

    images = os.listdir("{}/{}/".format(temp_storage_root,drug))

    vehicle_images = os.listdir("{}/{}/Vehicle".format(temp_storage_root,drug))
    
    organelle_slice = metadata.data[metadata.data['Structure']==organelle].reset_index(drop=True)
    plate_id = organelle_slice[organelle_slice['drug_label']==drug]['PlateId'].unique()[0]
    proto_row = organelle_slice[organelle_slice['drug_label']==drug].drop_duplicates(['PlateId']).drop(columns=['SourceFilename','drug_label'])
    row = 0
    
    for image in images:
        if image.endswith(".czi"):
            if str(plate_id) in image:
                proto_row['SourceFilename'] = "{}/{}/{}".format(temp_storage_root,drug,image)
                proto_row['drug_label']= drug
                organelle_metadata.data = pd.concat([organelle_metadata.data,proto_row])

                row+=1
            
    for image in vehicle_images:
        if image.endswith(".czi"):
            if str(plate_id) in image:
                proto_row['SourceFilename'] = "{}/{}/Vehicle/{}".format(temp_storage_root,drug,image)
                proto_row['drug_label'] = 'Vehicle'
                organelle_metadata.data = pd.concat([organelle_metadata.data,proto_row])
                row+=1    
    
    return organelle_metadata

def segment_and_create_dataset():
    print("Opening metadata file {}".format(datasets_metadata_dir))
    metadata  = DatasetMetadataSCV(datasets_metadata_dir,datasets_metadata_dir)
    print("Finish Opening metadata file {}...".format(datasets_metadata_dir))
    for organelle in organelles.keys():
        organelle_slice = metadata.data[metadata.data['Structure']==organelle]
        drugs = organelle_slice['drug_label'].unique()
        for drug in drugs:
            if drug != 'Vehicle':
                organelle_metadata = cretae_metadata_file(metadata,organelle,drug)
                count = 0
                threads = []
                combined_image_storage_path = []
                # organelle_metadata.data['combined_image_storage_path']="none"
                for i in range(organelle_metadata.get_shape()[0]):
                    fov_path=organelle_metadata.get_item(i,'SourceFilename')
                    fov_channel = organelle_metadata.get_item(i,'ChannelNumberBrightfield')-1
                    structure_fl_channel = organelle_metadata.get_item(i,'ChannelNumberStruct')-1
                    dna_fl_channel = organelle_metadata.get_item(i,'ChannelNumber405')-1 
                    mem_fl_channel = organelle_metadata.get_item(i,'ChannelNumber638')-1
                    drug_label = organelle_metadata.get_item(i,'drug_label')
                    new_image_path = "{}/{}_{}/{}_{}_{}.tiff".format(storage_root,organelle,drug,organelle,drug_label,i)
                    thread=threading.Thread(target=segment_and_create_image,args=(organelle,fov_path,fov_channel,structure_fl_channel,dna_fl_channel,mem_fl_channel,new_image_path))
                    combined_image_storage_path.append(new_image_path)                    
                    thread.start()
                    threads.append(thread)
                    count+=1;
                    if ((i+1)%num_threads == 0):
                        for thread in threads:
                            thread.join()
                            threads = []
                
                for thread in threads:
                    thread.join()
                organelle_metadata.data['combined_image_storage_path'] = combined_image_storage_path       
                organelle_metadata.create()
                print("Finish creating metadata csv: {}_{}".format(organelle,drug))            
                
                
def run():
    segment_and_create_dataset()
    
run()