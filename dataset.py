import shutil
import os
import threading
import tensorflow as tf
import tensorflow.keras as keras
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from cell_imaging_utils.image.image_utils import ImageUtils
from utils import *
import numpy as np
from tqdm import tqdm
import sklearn as sklearn
import cv2
from patchify import patchify
import init_env_vars

def get_size_in_GB(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size/(1024*1024*1024)

def mask_image_func(image_ndarray, mask_template_ndarray) -> np.ndarray:
    mask_ndarray = mask_template_ndarray
    return np.where(mask_ndarray == 255, image_ndarray, np.zeros(image_ndarray.shape))

##This class given the image list csv created in create_metadata script, will create a lazy loading dataset, this will work for single cells, FOVS, double input or double target. and even for classifiers
class DataGen(keras.utils.Sequence):

    def __init__(self, image_list_csv, 
                 input_col, 
                 target_col,
                 batch_size,
                 image_path_col = 'path_tiff',
                 num_batches=4,
                 patches_from_image=32,
                 patch_size=(16, 256, 256, 1),
                 
                 #use this to take partial list this will indicate where to start and where to finish
                 min_precentage=0, 
                 max_precentage=1,
                 
                 ## get a specific image number for the creation
                 image_number = None,
                 
                #the target will contain a tuple of the target_col and the input_col
                 input_as_y=False,
                #the input will contain a tuple of the target_col and the input_col
                 output_as_x=False,
                 
                 #crop single cell images by the bounding cube made by the whole cell segmentation
                 crop_edge=False,
                 #mask the single cell by the whole cell segmentation
                 mask=False,
                 #what column is contains the whole cell segmentation
                 mask_col='membrane_seg',
                #use dialtion on the mask
                 dilate=False,
                 dilate_kernel=np.ones((25, 25), np.uint8),
                 
                 #do normalization
                 norm=True,
                 #std or minmax
                 norm_type="std",
                 #add some noise to the input
                 noise=False,
                 #resize patch
                 resize=True,
                 #do rotation augmentation
                 augment=False,
                 #delete SSD cache from previous runs
                 delete_cahce=True,
                 
                 # a dict that map organelle to a model if predictors is True the output of the model will be joined to the input as the input
                 predictors=None,
                 
                 #If True the input will be pairs of the input_col and target_col and the target will by 1 if they are a matching pair and 0 otherwise
                 pairs=False,
                 #Mask them randomaly to make the prediction harder
                 masking_pair = False,
                 # if max value in a target patch of the pair, both the real target and the random one is lower then the cutoff value then we say both are backgrounfd and they are matching as well
                 cutoff = 0.0,
                 # % of matching pairs in the dataset this is a number > 1 (1:neg_ratio)
                 neg_ratio=1,

                 #True for outputing labels for clf
                 for_clf = False):

        self.new_path_origin = os.path.join(os.environ['DATA_PATH'], 'temp2')
        if delete_cahce:
            if os.path.exists(self.new_path_origin):
                shutil.rmtree(self.new_path_origin)

        self.df = DatasetMetadataSCV(image_list_csv, image_list_csv)
        self.n = self.df.get_shape()[0]
        self.input_col = input_col
        self.target_col = target_col
        self.image_path_col = image_path_col
        self.batch_size = batch_size
        self.num_batches = num_batches
        # if 1 resize image to patch size if >1 then sample patches with patch size
        self.patches_from_image = patches_from_image
        self.patch_size = patch_size
        self.min_precentage = min_precentage
        self.max_precentage = max_precentage
        self.image_number = image_number
        self.input_as_y = input_as_y  # output of target and input in Y
        self.output_as_x = output_as_x  # input of target and input in X
        self.crop_edge = crop_edge
        self.mask = mask
        self.mask_col = mask_col
        self.norm = norm
        self.dilate = dilate
        self.dilate_kernel = dilate_kernel
        self.noise = noise
        self.resize = resize
        self.augment = augment
        self.predictors = predictors
        self.pairs = pairs
        self.row_ratio = 4
        self.col_ratio = 4
        self.neg_ratio = neg_ratio
        self.masking_pair = masking_pair
        self.cutoff = cutoff
        self.norm_type = norm_type
        self.for_clf = for_clf
        self.patches_from_image = patches_from_image
        self.buffer_size = self.num_batches*self.batch_size*self.patches_from_image
        if self.pairs:
            self.buffer_size = self.buffer_size*self.row_ratio*self.col_ratio*(self.neg_ratio+1)
            self.input_buffer = np.zeros((self.buffer_size, self.patch_size[0],
                int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            self.target_buffer = np.zeros((self.buffer_size, self.patch_size[0],
                int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            # self.pred_buffer = np.zeros((self.buffer_size*self.row_ratio*self.col_ratio*(self.neg_ratio), self.patch_size[0],
            #     int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            self.labels_buffer = np.zeros(
                (self.buffer_size, 1))
        else:
            if (self.patch_size[0] == 1):  # 2D image
                two_d_patch_size = self.patch_size[1:]
                self.input_buffer = np.zeros(
                    (self.buffer_size, *two_d_patch_size))
                self.target_buffer = np.zeros(
                    (self.buffer_size, *two_d_patch_size))
            else:
                self.input_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
                self.target_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
                self.pred_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
            if self.for_clf:
                self.labels_buffer = np.zeros((self.buffer_size, 1))
        try:
            if (not os.path.exists(self.new_path_origin)):
                os.makedirs(self.new_path_origin)
        except Exception as e:
                print("SSD storage is not exist in {}".format(self.new_path_origin))
                print(e)
                self.new_path_origin = os.path.join(os.environ['DATA_PATH'], 'temp/{}'.format(os.environ['LOGNAME']))
                if (not os.path.exists(self.new_path_origin)):
                    os.makedirs(self.new_path_origin)

        self.fill_buffer()

    def get_image_from_ssd(self, file_path, prefix):
        file_name = file_path.split('/')[-1]
        new_file_path = "{}/{}_{}".format(
            self.new_path_origin, prefix, file_name)
        if os.path.exists(new_file_path):
            image_ndarray = ImageUtils.image_to_ndarray(
                ImageUtils.imread(new_file_path))
            return image_ndarray, new_file_path
        return None, new_file_path

    def on_epoch_end(self):
        self.fill_buffer()

    def fill_samples(self, i):
        j = 0
        count = 0
        while j < self.batch_size and count<30:
            count += 1
            image_index = None
            try:
                if self.image_number is None:
                    image_index = np.random.randint(
                        int(self.n*self.min_precentage), int(self.n*self.max_precentage))
                    # print("adding image {} to buffer\n".format(image_index))
                else:
                    image_index = self.image_number

                image_path = self.df.get_item(image_index, self.image_path_col)

                if (self.augment):
                    k = np.random.random_integers(0, 3)
                else:
                    k = 0

                input_image, new_input_path = self.get_image_from_ssd(
                    image_path, self.input_col)
                if (self.augment and input_image is not None):
                    # print(input_image.shape)
                    input_image = np.rot90(input_image, axes=(2,3), k=k)
                    
                if not self.for_clf:
                    target_image, new_target_path = self.get_image_from_ssd(
                        image_path, self.target_col)
                    if (self.augment and target_image is not None):
                        target_image = np.rot90(target_image, axes=(2,3), k=k)
                if self.predictors is not None:
                    pred_image, new_prediction_path = self.get_image_from_ssd(
                        image_path, "prediction")
                    if (self.augment and pred_image is not None):
                        pred_image = np.rot90(pred_image, axes=(2,3), k=k)

                if (input_image is None or (target_image is None and not self.for_clf)):
                    image_ndarray = ImageUtils.image_to_ndarray(
                        ImageUtils.imread(image_path))
                    image_ndarray = image_ndarray.astype(np.float32)
                    if (self.augment):
                        image_ndarray = np.rot90(image_ndarray, axes=(2, 3), k=k)
                    if (self.crop_edge):
                        mask_index = int(self.df.get_item(
                            image_index, self.mask_col))
                        mask_template = ImageUtils.get_channel(
                            image_ndarray, mask_index)
                        image_ndarray = ImageUtils.crop_edges(
                            image_ndarray, mask_template)

                    mask_image = None
                    if (self.mask):
                        mask_index = int(self.df.get_item(
                            image_index, self.mask_col))
                        mask_image = ImageUtils.get_channel(
                            image_ndarray, mask_index)
                        if (self.dilate):
                            for h in range(mask_image.shape[0]):
                                mask_image[0, h, :, :] = cv2.dilate(mask_image[0, h, :, :].astype(
                                    np.uint8), self.dilate_kernel)

                    channel_index = int(self.df.get_item(
                        image_index, self.input_col))
                    input_image = ImageUtils.get_channel(
                        image_ndarray, channel_index)
                    if (self.mask):
                        input_image = mask_image_func(input_image, mask_image)
                    if (self.for_clf):
                        mask_index = int(self.df.get_item(image_index, 'structure_seg'))
                        masked_input_image = ImageUtils.get_channel(image_ndarray, mask_index)      
                    if (self.norm):
                        if self.norm_type == "std":
                            input_image = ImageUtils.normalize_std(input_image)
                        else:
                            input_image = ImageUtils.normalize(input_image, max_value=1.0, dtype=np.float32)

                    if (self.target_col == self.input_col or self.for_clf):
                        target_image = input_image
                    else:
                        channel_index = int(self.df.get_item(
                            image_index, self.target_col))
                        target_image = ImageUtils.get_channel(
                            image_ndarray, channel_index)
                        if (self.mask):
                            target_image = mask_image_func(
                                target_image, mask_image)
                        if (self.dilate):
                            for h in range(target_image.shape[1]):
                                target_image[0, h, :, :] = cv2.dilate(target_image[0, h, :, :].astype(
                                    np.uint8), self.dilate_kernel)                            
                        else:
                            if (self.norm):
                                if self.norm_type == "std":
                                    target_image = ImageUtils.normalize_std(target_image)
                                else:
                                    target_image = ImageUtils.normalize(target_image, max_value=1.0, dtype=np.float32)
                    if (self.noise):
                        target_image += np.random.normal(0,
                                                        0.05, size=target_image.shape)
                        target_image = np.clip(target_image, 0, 1)
                    # input_image = np.expand_dims(input_image[0], axis=-1)
                    # target_image = np.expand_dims(target_image[0], axis=-1)
                    if get_size_in_GB(self.new_path_origin)<100:
                        ImageUtils.imsave(input_image, new_input_path)
                    if not self.for_clf:
                        if get_size_in_GB(self.new_path_origin)<100:
                            ImageUtils.imsave(target_image, new_target_path)

                if self.pairs:
                    input_image = ImageUtils.to_shape(input_image, self.patch_size)
                    target_image = ImageUtils.to_shape(target_image, self.patch_size)
                    input_patches = patchify(input_image, (self.patch_size[0], int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio),self.patch_size[-1]), step=int(self.patch_size[2]/self.col_ratio))
                    target_patches = patchify(target_image, (self.patch_size[0], int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio),self.patch_size[-1]), step=int(self.patch_size[2]/self.col_ratio))
                    ### add masking
                    mask_vol = 4
                    mask_ratio = 0.75 #0.75
                    num_masks = int((np.prod(input_patches.shape[3:-1])*mask_ratio)/mask_vol)
                    masking_ind_z = np.random.random_integers(0,int(self.patch_size[0]/mask_vol)-1,num_masks)
                    masking_ind_x = np.random.random_integers(0,int(self.patch_size[1]/mask_vol)-1,num_masks)
                    masking_ind_y = np.random.random_integers(0,int(self.patch_size[2]/mask_vol)-1,num_masks)
                    if self.masking_pair:
                        masked_target_patchs = np.copy(target_patches)
                        masked_input_patchs = np.copy(input_patches)
                        for w in range(num_masks):
                            # masked_target_patchs[:,:,:,:,masking_ind_z[w]*mask_vol:(masking_ind_z[w]+1)*mask_vol,masking_ind_x[w]*mask_vol:(masking_ind_x[w]+1)*mask_vol,masking_ind_y[w]*mask_vol:(masking_ind_y[w]+1)*mask_vol,0] = 0
                            masked_input_patchs[:,:,:,:,masking_ind_z[w]*mask_vol:(masking_ind_z[w]+1)*mask_vol,masking_ind_x[w]*mask_vol:(masking_ind_x[w]+1)*mask_vol,masking_ind_y[w]*mask_vol:(masking_ind_y[w]+1)*mask_vol,0] = 0
                    else:
                        masked_target_patchs = target_patches
                        masked_input_patchs = input_patches
                    for l in range(input_patches.shape[1]):
                        for m in range(input_patches.shape[2]):
                            pos = i*(self.batch_size*input_patches.shape[1]*input_patches.shape[2]*(self.neg_ratio+1))+j*(input_patches.shape[1]*input_patches.shape[2]*(self.neg_ratio+1))+l*(input_patches.shape[2]*(self.neg_ratio+1))+m*(self.neg_ratio+1)
                            self.input_buffer[pos] = masked_input_patchs[0,l,m]
                            self.target_buffer[pos] = masked_target_patchs[0,l,m]
                            self.labels_buffer[pos] = 1
                            for nr in range(1,self.neg_ratio+1):
                                l_random = np.random.random_integers(0,input_patches.shape[1]-1)
                                m_random = np.random.random_integers(0,input_patches.shape[2]-1)
                                neg_label = 0
                                if (np.max(target_patches[0,l_random,m_random])<=self.cutoff and  np.max(target_patches[0,l,m])<=self.cutoff):
                                    neg_label = 1
                                self.input_buffer[pos+nr] = masked_input_patchs[0,l,m]
                                self.target_buffer[pos+nr] = masked_target_patchs[0,l_random,m_random]
                                self.labels_buffer[pos+nr] = neg_label
                                
                else:
                    if (self.patches_from_image > 1):
                        # Sample patches
                        input_image = np.moveaxis(input_image, 0,-1)
                        target_image = np.moveaxis(target_image, 0,-1)
                        # input_image = ImageUtils.to_shape(input_image,input_image.shape,None, min_shape=self.patch_size)
                        # target_image = ImageUtils.to_shape(target_image,input_image.shape,None, min_shape=self.patch_size)
                        for k in range(self.patches_from_image):
                            patch_indexes = []
                            for o in range(len(self.patch_size)-1):
                                random_patch_index = np.random.randint(
                                    input_image.shape[o]-self.patch_size[o]+1)
                                patch_indexes.append(
                                    (random_patch_index, random_patch_index+self.patch_size[o]))
                            input_patch = ImageUtils.slice_image(
                                input_image, [*patch_indexes, (None, None)])
                            target_patch = ImageUtils.slice_image(
                                target_image, [*patch_indexes, (None, None)])
                            if (self.patch_size[0] == 1):  # 2D image
                                input_patch = input_patch[0]
                                target_patch = target_patch[0]
                            pos = i*(self.batch_size*self.patches_from_image)+j*(self.patches_from_image)+k
                            self.input_buffer[pos] = input_patch
                            self.target_buffer[pos] = target_patch
                            if self.for_clf:
                                masked_input_patch = ImageUtils.slice_image(masked_input_image, [*patch_indexes, (None, None)])                                
                                label = 12
                                if np.sum(masked_input_patch/255.) > 1:
                                    label = self.df.get_item(image_index, 'label')
                                self.labels_buffer[pos] = label
                    else:
                        if (self.resize):
                            # Resize to patch size
                            # input_image = ImageUtils.to_shape(
                            #     input_image, (16, *self.patch_size[1:]))
                            # target_image = ImageUtils.to_shape(
                            #     target_image, (16, *self.patch_size[1:]))
                            input_image = ImageUtils.to_shape(
                                input_image, (16,256,256,1))
                            target_image = ImageUtils.to_shape(
                                target_image, (16,256,256,1))
                            # ndimage.zoom(input_image,[self.patch_size[0]/input_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)
                            input_image = resize_image(
                                self.patch_size, input_image)
                            # ndimage.zoom(target_image,[self.patch_size[0]/target_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)
                            target_image = resize_image(
                                self.patch_size, target_image)
                        else:
                            input_image = ImageUtils.to_shape(
                                input_image, self.patch_size)
                            target_image = ImageUtils.to_shape(
                                target_image, self.patch_size)
                            if self.predictors is not None:
                                if pred_image is None and self.predictors is not None:
                                    if isinstance(self.predictors,dict):
                                        organelle = image_path.split('/')[-1]
                                        organelle = organelle.split("_")[0]
                                        pred_image = self.predictors[organelle](np.expand_dims(input_image, axis=0)).numpy()

                                    else:
                                        channel_index = int(self.df.get_item(image_index, "channel_dna"))
                                        pred_image = ImageUtils.get_channel(image_ndarray, channel_index)
                                    if (self.norm):
                                        if self.norm_type == "std":
                                            pred_image = ImageUtils.normalize_std(pred_image)
                                        else:
                                            pred_image = ImageUtils.normalize(pred_image, max_value=1.0, dtype=np.float32)  
                                    if get_size_in_GB(self.new_path_origin)<100:
                                        ImageUtils.imsave(pred_image, new_prediction_path)                                  
                                    self.pred_buffer[i*self.batch_size+j *self.patches_from_image] = pred_image

                        self.input_buffer[i*self.batch_size+j *
                                        self.patches_from_image] = input_image
                        self.target_buffer[i*self.batch_size+j *
                                        self.patches_from_image] = target_image
                        if self.for_clf:
                            label = self.df.get_item(image_index, 'label')
                            self.labels_buffer[i*self.batch_size+j * self.patches_from_image] = label
                j+=1
                count = 0
            except Exception as e:
                try:
                    print("could not load image {} pi:{} - {}, trying another one.\n".format(image_path,input_image.shape,self.patch_size))
                    print(str(e))
                except:
                    print(str(e))

            # if (old_image_path != image_path):
            #     os.remove(image_path)
          
    def fill_buffer(self):
        threads = []
        num_threads = 4
        for i in tqdm(range(self.num_batches)):
            if (num_threads > 1):
                thread = threading.Thread(target=self.fill_samples, args=[i])
                thread.start()
                threads.append(thread)

                if ((i+1) % num_threads == 0):
                    for t in (range(len(threads))):
                        threads[t].join()
                    threads = []
            else:
                self.fill_samples(i)

        for thread in threads:
            thread.join()
        if  self.pairs:
            self.input_buffer, self.target_buffer, self.labels_buffer = sklearn.utils.shuffle(
            self.input_buffer, self.target_buffer, self.labels_buffer)
        else:
            self.input_buffer, self.target_buffer = sklearn.utils.shuffle(
            self.input_buffer, self.target_buffer)

    def __getitem__(self, index):
        num_samples = int(self.buffer_size/(self.num_batches))
        if (self.patches_from_image > 1):
            num_samples = int(self.buffer_size/(self.num_batches*self.patches_from_image))
        if self.pairs:
            num_samples = int(self.buffer_size/(self.num_batches*self.col_ratio))
        X = self.input_buffer[index*num_samples:(index+1)*num_samples]
        # y1 = deepcopy(self.input_buffer[index*self.batch_size:(index+1)*self.batch_size])
        
        Y = self.target_buffer[index*num_samples:(index+1)*num_samples]
        if (self.input_as_y):
            Y = [X, Y]
        if (self.output_as_x):
            X = [X, Y]
        if self.predictors is not None:
            P = self.pred_buffer[index*num_samples:(index+1)*num_samples]
            X = [X, P]
        if self.pairs:
            L = self.labels_buffer[index*num_samples:(index+1)*num_samples]
            X = [X,Y]
            Y = L
        if self.for_clf:
            X = X
            Y = self.labels_buffer[index*num_samples:(index+1)*num_samples]
        return X, Y

    def __len__(self):
        if (self.patches_from_image > 1):
            return self.num_batches*self.patches_from_image
        if self.pairs:
            return self.num_batches*self.col_ratio
        return self.num_batches
