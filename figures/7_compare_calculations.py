import global_vars as gv
from gui.gui_logic import *
from PIL import Image
import matplotlib.pyplot as plt
from figure_config import figure_config
import os
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


params = [
          {"organelle":"Nucleolus\n(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5,"unet":"../unet_model_22_05_22_ngc_128"},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0,"unet":"../unet_model_22_05_22_ne_128"},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5,"unet":"../unet_model_22_05_22_mito_128"},
          {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5,"unet":"../unet_model_22_05_22_membrane_128"},
          ]
gv.patch_size = (32,128,128,1)


#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"
base_dir = os.path.join(gv.BASE_PATH, 'full_cells_fovs')
X_gradcam = False
layer_name = "unet_convt_bottleneck2"

# Create a single row with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(5, 10))  # Adjust the figure size as needed
i = 0
for param in params:
  gv.model_path = param["model"]
  gv.interpert_model_path = param["unet"]
  gv.organelle = param["organelle"]
  dataset_path = "{}/train_test_list/{}/image_list_test.csv".format(base_dir,param["organelle"])
  dataset = get_dataset(dataset_path)
  noise_scale = param["noise"]
  unet_model = load_model(param["unet"])
  mg_model = load_model(param["model"])

  selected_layer = unet_model.get_layer(layer_name)
  evaluate_interperters(param["model"],dataset,unet_model,mg_model,selected_layer,X_gradcam,noise_scale = noise_scale)    
  png_filename = "{}/comparison.png".format(param["model"])
  legend = False
  # if i==3:
  #   legend = True
  ylabel = False
  if i==3:
        ylabel = True
  plot_evaluation_graph_std(png_filename, ["saliency","gradcam","mask_interperter"], legend,ylabel)
  im_graph = Image.open(png_filename)
  
  axes[i].imshow(im_graph)
  axes[i].axis('off')
  axes[i].set_title(param['organelle'], fontsize=10, fontname=figure_config["font"])
  i += 1
  
# Display the plot
# Adjust the layout to change spaces between images
fig.subplots_adjust(wspace=0.0, hspace=0.0)  # Change horizontal/vertical spacing

# Display the plot
# fig.tight_layout(pad=0)  # Additional adjustment to tighten the layout
fig.savefig("../figures/compare_method.png", bbox_inches='tight', pad_inches=0.01)
