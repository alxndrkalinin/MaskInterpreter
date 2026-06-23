import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import tensorflow_addons as tfa
import global_vars as gv

from utils.metrics import tf_pearson_corr
tf.compat.v1.enable_eager_execution()

"""Mask Interpreter"""
class MaskInterpreter(keras.Model):

    def __init__(self, patch_size, adaptor, unet, weighted_pcc,pcc_target=0.9, **kwargs):
        """
        Args:
            patch_size (list): patch size
            adaptor (model): # A model that will create the mask, it's input is a conv layer result with 64 channels of the input and prediction of the interpert model
            unet (model): The interpret model
            weighted_pcc (Bool): Use regular or weighted PCC for the target score term
            pcc_target (float, optional): The target score. Defaults to 0.9.
        """
        super(MaskInterpreter, self).__init__(**kwargs)

        self.weighted_pcc = weighted_pcc
        self.pcc_target = pcc_target
        self.unet = unet
        
        image_input = keras.layers.Input(shape=patch_size,dtype=tf.float32)
        target_input = keras.layers.Input(shape=patch_size,dtype=tf.float32)
        processed_target = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(target_input)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        processed_input = keras.layers.Concatenate(axis=-1)([processed_image,processed_target])
        importance_mask = tf.cast(adaptor(processed_input),dtype=tf.float64)
        self.generator = keras.Model([image_input,target_input],importance_mask,name="generator")
        
        #Measure the similiarity of the predictions of the original input and the moisy input
        self.similiarity_loss_tracker = keras.metrics.Mean(
            name="similiarity_loss"
        )
        
        #Measure the % of pixels in the importance mask that is smaller then 0.5 (0 is not important for the prediction and 1 is important)
        self.binary_size_mask = keras.metrics.BinaryAccuracy(
            name="binary_size_mask"
        )
        
        #Measure the size of the importance mask in % (1 means that every pixel is important, 0 no pixel)
        self.importance_mask_size = keras.metrics.Mean(
            name="importance_mask_size"
        )
        
        #Total loss term
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        
        #Measure the PCC between the predictions of the noisy and the original inputs
        self.pcc = keras.metrics.Mean(name = "pcc")
        
        #Show the stop value which decides when to stop the training, it is the linear composition of the distance from the target score and the size of the mask self.pcc_target-pcc_loss + mean_mask
        self.stop = keras.metrics.Mean(name = "stop")
        
    def compile(self, g_optimizer, similiarity_loss_weight=1. ,mask_loss_weight=1.0, noise_scale=1.5, target_loss_weight=10.0, run_eagerly=False):
        """Compile the model

        Args:
            g_optimizer (float): learining rate
            similiarity_loss_weight (float, optional): factor to multiply the similiarity loss term, usually the MSE. Defaults to 1..
            mask_loss_weight (float, optional): factor to multiply the importance mask loss term, usually the SUM of the importance values. Defaults to 0.01.
            noise_scale (float, optional): the STD of the noise generated, usually chosen by noising the entire input and measuring the predictions similarity. Defaults to 5.0.
            target_loss_weight(float, optional): the weight for the PCC target term -> if 0.0 means the loss will not use it.
            run_eagerly (bool, optional): keras args. Defaults to False.
        """
        super(MaskInterpreter, self).compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.similiarity_loss_weight = similiarity_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.noise_scale = noise_scale
        self.target_loss_weight = target_loss_weight
        print(f"compile: g_optimizer:{g_optimizer}, similiarity_loss_weight:{similiarity_loss_weight}, mask_loss_weight:{mask_loss_weight}, noise_scale:{noise_scale}, target_loss_weight:{target_loss_weight}")
    
    @property
    def metrics(self):
        return [
            self.similiarity_loss_tracker,
            self.total_loss_tracker,
            self.binary_size_mask,
            self.importance_mask_size,
            self.pcc,
            self.stop,
        ]

    def train_step(self, data, train=True):
        data_0 = tf.cast(data[0],dtype=tf.float64)
        data_1 = tf.cast(data[1],dtype=tf.float64)
        
        unet_target = self.unet(data_0)
        
        with tf.GradientTape() as tape:
            importance_mask = self.generator([data_0,unet_target])
            
            normal_noise = tf.random.normal(tf.shape(importance_mask),stddev=self.noise_scale,dtype=tf.float64) 

            adapted_image = (importance_mask*data_0)+(normal_noise*(1-importance_mask))
            
            unet_predictions = self.unet(adapted_image)

            similiarity_loss = tf.reduce_mean(
               tf.reduce_mean(
                #    keras.losses.mean_absolute_error(unet_target, unet_predictions),axis=(1,2)
                   keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(1,2)
               ),axis=(0,1)
            )            
            
            mean_importance_mask = tf.reduce_mean((importance_mask))

            importance_mask_loss = tf.reduce_mean(
               tf.reduce_mean(
                  keras.losses.mean_squared_error(tf.zeros_like(importance_mask), importance_mask),axis=(1,2)
                #   keras.losses.mean_absolute_error(tf.zeros_like(importance_mask), importance_mask),axis=(1,2)
               ),axis=(0,1)
            )
            
            if self.weighted_pcc:
                #data_1 is the segmented organelle for the weighted PCC
                human_eval = (tf_pearson_corr(unet_target,unet_predictions,data_1))
            else:
                human_eval = (tf_pearson_corr(unet_target,unet_predictions))
            
            pcc_loss =  tf.math.abs(self.pcc_target-tf.minimum(human_eval,self.pcc_target))
            
            total_loss = (similiarity_loss)*self.similiarity_loss_weight + (importance_mask_loss)*self.mask_loss_weight + (pcc_loss)*self.target_loss_weight
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
        
        self.importance_mask_size.update_state(mean_importance_mask)
        self.similiarity_loss_tracker.update_state(similiarity_loss)  
        self.binary_size_mask.update_state(tf.zeros_like(importance_mask), importance_mask)
        self.pcc.update_state(human_eval)
        self.total_loss_tracker.update_state(total_loss)
        self.stop.update_state(pcc_loss + importance_mask_loss)
               
        return {
            "similiarity_loss": self.similiarity_loss_tracker.result(),
            "binary_size": self.binary_size_mask.result(),
            "importance_mask_size": self.importance_mask_size.result(),
            "total_loss":self.total_loss_tracker.result(),
            "pcc":self.pcc.result(),
            "stop": self.stop.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        original_unet = self.unet(input)
        return self.generator([input,original_unet])