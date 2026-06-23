import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import math
from figure_config import figure_config
import global_vars as gv
# Define the path to your CSV files
path = os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv')

columns_to_plot = [['drug_label','time_point (hr)',]]
# List of columns with the results to plot
params = [
          {"organelle":"Endoplasmic reticulum","model":"../unet_model_22_05_22_er_128","target":"channel_target","calculate_on":"Endoplasmic reticulum"},
          {"organelle":"Plasma-membrane","model":"../unet_model_22_05_22_membrane_128","target":"channel_membrane","calculate_on":None},
          {"organelle":"DNA","model":"../unet_model_22_05_22_dna_128","target":"channel_dna","calculate_on":None},
          {"organelle":"Golgi","model":"../unet_model_22_05_22_golgi_128","target":"channel_target","calculate_on":'Golgi'},
          {"organelle":"Actomyosin bundles","model":"../unet_model_22_05_22_bundles_128","target":"channel_target","calculate_on":'Actomyosin bundles'},
          {"organelle":"Microtubules","model":"../unet_model_22_05_22_microtubules_128","target":"channel_target","calculate_on":'Microtubules'},
          {"organelle":"Actin filaments","model":"../unet_model_22_05_22_actin_128","target":"channel_target","calculate_on":'Actin filaments'},
          ]

# Function to plot box plots in a dynamic grid layout based on the number of params
def plot_box_plots(data, columns, params):
    colors = ['lightblue', 'green', 'blue', 'lightyellow', 'lightgray', 'lightyellow', 'lightgoldenrodyellow', 
              'lightsteelblue', 'lavender', 'honeydew', 'gray', 'green', 'yellow', 'orange', 'pink', 'brown']
    
    for x_col in tqdm(columns):
        num_plots = len(params)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)  # Dynamically calculate rows based on 3 columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(64, num_rows * 6))  # Adjust figure size based on rows
        fig.suptitle(f'UNET predictions vs GT by {x_col}', fontsize=figure_config["title"], fontname=figure_config["font"])
        plt.subplots_adjust(hspace=0.3, wspace=0.1,top=0.8)
        
        for idx, param in enumerate(params):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case
                
            control_df = pd.read_csv("{}/pcc_resuls.csv".format(param["model"]))
            
            data_no_control = data
            unique_values = data_no_control[x_col].drop_duplicates().values
            groups = []
            for val in unique_values:
                group = data_no_control[(data_no_control[x_col].values == list(val))][param["model"]].dropna()
                group = group[group.values>0.2]
                groups.append(group)
            control_values = control_df['PCC'].dropna()
            
            # Calculate fold change and p-values
            fold_changes = [np.mean(group) / np.mean(control_values) for group in groups]
            p_values = []
            for group in groups:
                if len(group) > 0:
                    p_values.append(mannwhitneyu(group, control_values, alternative='less').pvalue )
                else:
                    p_values.append(1)
            # Add control values to the groups
            groups.append(control_values)
            labels = [f'{val}' for i, val in enumerate(unique_values)]
            labels.append(f'Test set')
            
            y_col = param["model"][3:]
            box = ax.boxplot(groups, labels=labels, patch_artist=True, vert=True, showfliers=False)
            
            # Color the box plots
            for patch, color in zip(box['boxes'], colors[:len(groups)]):
                patch.set_facecolor(color)
            
            ax.set_ylim(0.15, max(np.max(control_values), max([group.max() for group in groups])) + 0.1)
            ax.set_title(f"{param['organelle']}",fontsize=figure_config["organelle"], fontname=figure_config["font"])
            ax.set_xlabel(x_col,fontsize=figure_config["text"], fontname=figure_config["font"])
            if col == 0:
                ax.set_ylabel('UNET prediction vs GT [PCC]',fontsize=figure_config["axis"], fontname=figure_config["font"])
            # else:
                # Turn off y-axis
                # ax.get_yaxis().set_visible(False)
            
            ax.tick_params(axis='y', rotation=90)
            ax.tick_params(axis='x', rotation=0)

            # Annotate fold change and p-values
            for i, (fold_change, p_value) in enumerate(zip(fold_changes, p_values)):
                text_y = max(np.max(control_values), max([group.max() for group in groups])) + 0.05
                if p_value < 0.01 and np.median(groups[i]) < np.median(control_values):
                    ax.text(i + 1,text_y+0.14, '**', ha='center', va='top', color='blue', fontsize=14, fontname=figure_config["font"])
                label = f"FC={fold_change:.2f}\np-value={p_value:.2g}"
                ax.text(i + 1,text_y+0.075, label, ha='center', va='top', fontsize=figure_config["text"], fontname=figure_config["font"])
        
        # Remove any empty subplots
        for idx in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])
        
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(gv.CWD, f'figures/{x_col}_comparison_unet_pertrub.png'),bbox_inches='tight',pad_inches=0.05)
        plt.close()

# Plot box plots for each specified column and result column
plot_box_plots(pd.read_csv(path), columns_to_plot, params)
