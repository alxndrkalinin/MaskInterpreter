import os
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from utils.callbacks import *
from utils.metrics import *
import pandas as pd
from utils.utils import *

CONTINUE_TRAINING = False ## False to override current model with the same name

#Model to train
gv.model_type = "MG"

#If Mask Interpreter then add the path to the model you want to interpret
gv.interpert_model_path = "../unet_model_22_05_22_dna_128" ## UNET model if in MG mode it is the model that we want to interpret
#path to the model
gv.model_path = "../mg_model_dna_13_05_24_1.5_new" ## the model will be saved here

#Input and target channels in the image
gv.input = "channel_signal"
gv.target = "channel_dna"

#Organelle to train the model upon
gv.organelle = "Nucleolus-(Granular-Component)" #"Actomyosin-bundles"#"Golgi" #"Plasma-membrane" #"Microtubules" #"Actin-filaments" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)" #"Tight-junctions" #"Endoplasmic-reticulum" 

#Assemble the proper tarining csvs by the organelle, model type, and if the data is pertrubed or not
gv.train_ds_path = os.path.join(gv.DATA_PATH, "{}/image_list_train.csv".format(gv.organelle))
gv.test_ds_path = os.path.join(gv.DATA_PATH, "{}/image_list_test.csv".format(gv.organelle))

#if compound is not None then it will take pertrubed dataset
compound = None #"s-Nitro-Blebbistatin" #"s-Nitro-Blebbistatin" #"Staurosporine" #None #"s-Nitro-Blebbistatin" #None #"paclitaxol_vehicle" #None #"paclitaxol_vehicle" #"rapamycin" #"paclitaxol" #"blebbistatin" #""
#drug could be either the compound or Vehicle which is like DMSO (the unpertrubed data in the pertrubed dataset)
drug = compound #"Vehicle"
if compound is not None:
    ds_path = os.path.join(gv.CWD, "single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(gv.organelle,compound,drug))
else:
    ds_path = gv.train_ds_path
    
gv.batch_size = 4
norm_type = "std"
gv.patch_size = (32,128,128,1)

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
print(gv.organelle)
#example to add predictors to the dataset predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet}
train_dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.8,augment=True,norm_type=norm_type, predictors=None,delete_cahce=True)
validation_dataset = DataGen(ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 16, patch_size=gv.patch_size,min_precentage=0.8,max_precentage=1.0,augment=False,norm_type=norm_type, predictors=None)

if (gv.model_type == "UNET"):
    from models.UNETO import *
    num_channels = 1
    model = get_unet((gv.patch_size[0],gv.patch_size[1],gv.patch_size[2],num_channels),activation="linear")
    model.summary()
    
    model = UNET(model,num_channels)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),run_eagerly=True)
    
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        unet_pt = keras.models.load_model(gv.model_path)                           
        model.set_weights(unet_pt.get_weights())
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=150,monitor="val_loss",restore_best_weights=True)
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])

elif (gv.model_type == "MG"):
    from models.MaskInterpreter import *
    from models.UNETO import *
    similiarity_loss_weight = 1.0  # 1.0 default
    target_loss_weight = 10.0 #10.0 is the default value 
    mask_loss_weight=1.0 #1.0 is the default value 
    noise_scale = 1.5 #value according to find_noise_scale
    
    #The default target score calculation is regular PCC, if one wish to use weighted PCC uncomment the line below
    weighted_pcc = False
    #Uncomment below for modified pearson
    # weighted_pcc = True
    
    
    if weighted_pcc:
        dilate = weighted_pcc #False for regular pearson, True for modified
        gv.target = "structure_seg"
        train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.8,augment=True,norm_type=norm_type, dilate=dilate) 
        validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.8,max_precentage=1.0,augment=False,norm_type=norm_type, dilate=dilate)
    
    interpert_model = keras.models.load_model(gv.interpert_model_path)
    interpert_model.summary()
    
    # A model that will create the mask, it's input is a conv layer result with 64 channels of the input and prediction of the interpert model output the importance mask
    adaptor = get_unet((*gv.patch_size[:-1],64),activation="sigmoid") 
    adaptor.summary()
    
    model = MaskInterpreter(gv.patch_size, adaptor, interpert_model, weighted_pcc=weighted_pcc,pcc_target=0.9)
    model.unet.trainable = False
    
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        mg_pt = keras.models.load_model(gv.model_path)
        model.set_weights(mg_pt.get_weights())
    
    # checkpoint callback monitoring "val_stop" decides when to save the epoch, it is the linear composition of the distance from the target score and the size of the mask self.pcc_target-pcc_loss + mean_mask
    # term and term value make surm that the value of term in the loss is greater then the term value before saving the epoch
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path,monitor="val_stop",term="val_pcc",term_value=0.03)
    print("similiarity_loss_weight: ", similiarity_loss_weight)
    print("mask_loss_weight: ",mask_loss_weight)
    print("noise_scale: ",noise_scale)
    print("target_loss_weight:",target_loss_weight)
    
    early_stop_callback = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_total_loss")
    # weight_loss_adaptor_callback = ChangeWeightLossCallbackMaskInterpreter()
    model.compile(g_optimizer = keras.optimizers.Adam(learning_rate=0.0001),mask_loss_weight=mask_loss_weight,noise_scale=noise_scale,target_loss_weight=target_loss_weight,similiarity_loss_weight=similiarity_loss_weight,run_eagerly=True)
    
    # model.save("./model.h5")
    
    losses = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[checkpoint_callback,early_stop_callback])  #weight_loss_adaptor_callback
        
    a = pd.DataFrame(losses.history)
    create_dir_if_not_exist(gv.model_path)
    a.to_csv("{}/losses.csv".format(gv.model_path))

model.save(gv.model_path,save_format="tf")