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
path = os.path.join(gv.DATA_PATH, 'unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv')

columns_to_plot = ['Workflow']
# List of columns with the results to plot
params = [
    {"organelle":"DNA","model":"../unet_model_22_05_22_dna_128"},
    # {"organelle":"Plasma-membrane","model":"../unet_model_22_05_22_membrane_128"},
]

# Function to plot box plots in a dynamic grid layout based on the number of params
def plot_box_plots(data, columns, params):
    colors = ['lightblue', 'green', 'blue', 'lightyellow', 'lightgray', 'lightyellow', 'lightgoldenrodyellow', 
              'lightsteelblue', 'lavender', 'honeydew', 'gray', 'green', 'yellow', 'orange', 'pink', 'brown']
    
    for x_col in tqdm(columns):
        num_plots = len(params)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)  # Dynamically calculate rows based on 3 columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 6))  # Adjust figure size based on rows
        fig.suptitle(f'UNET predictions vs GT by {x_col}', fontsize=figure_config["title"], fontname=figure_config["font"])
        plt.subplots_adjust(hspace=0.3, wspace=0.1,top=0.8)
        
        for idx, param in enumerate(params):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case
            
            if param["organelle"] == "DNA":
                control_organelle = "Nucleolus-(Granular-Component)"
            else:
                control_organelle = param["organelle"]
                
            train_test_df = pd.read_csv(path)
            test_df = pd.read_csv(os.path.join(gv.BASE_PATH, f'full_cells_fovs/train_test_list/{control_organelle}/image_list_test.csv'))
            control_df = pd.merge(train_test_df, test_df['path_tiff'], how='inner', left_on='combined_image_storage_path', right_on='path_tiff')
            control_structure = control_df['StructureShortName'].values[0]
            
            data_no_control = data[data['StructureShortName'] != control_structure]
            unique_values = np.sort(data_no_control[x_col].unique())
            groups = [data_no_control[data_no_control[x_col] == val][param["model"]].dropna() for val in unique_values]
            control_values = control_df[param["model"]].dropna()
            
            # Calculate fold change and p-values
            fold_changes = [np.median(group) / np.median(control_values) for group in groups]
            p_values = [mannwhitneyu(group, control_values, alternative='less').pvalue for group in groups]
            
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
            #     # Turn off y-axis
            #     ax.get_yaxis().set_visible(False)
            
            ax.tick_params(axis='y', rotation=90)
            ax.tick_params(axis='x', rotation=0)

            # Annotate fold change and p-values
            # for i, (fold_change, p_value) in enumerate(zip(fold_changes, p_values)):
            #     text_y = max(np.max(control_values), max([group.max() for group in groups])) + 0.05
            #     if p_value < 0.01 and np.median(groups[i]) < np.median(control_values):
            #         ax.text(i + 1,text_y+0.14, '**', ha='center', va='top', color='blue', fontsize=14, fontname=figure_config["font"])
            #     label = f"FC={fold_change:.2f}\np-value={p_value:.2g}"
            #     ax.text(i + 1,text_y+0.075, label, ha='center', va='top', fontsize=figure_config["text"], fontname=figure_config["font"])
        
        # Remove any empty subplots
        for idx in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])
        
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(gv.CWD, f'figures/{x_col}_comparison_unet2.png'),bbox_inches='tight',pad_inches=0.05)
        plt.close()

# Plot box plots for each specified column and result column
plot_box_plots(pd.read_csv(path), columns_to_plot, params)
