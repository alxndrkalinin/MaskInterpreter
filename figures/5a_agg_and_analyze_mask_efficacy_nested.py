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
path = os.path.join(gv.DATA_PATH, '**/image_list_with_metadata__with_efficacy_scores_full.csv')

# Print the path pattern
print(f"Using path pattern: {path}")

# Use glob to get all the CSV file paths
csv_files = glob.glob(path, recursive=True)

# Print the found files
print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(file)

# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Check if any DataFrames were created
if not dfs:
    print("No DataFrames were created. Please check the CSV files.")
else:
    # Concatenate all DataFrames
    metadata_with_efficacy_scores_df = pd.concat(dfs, ignore_index=True)
    print("metadata_with_efficacy_scores_df # FOVS:{}".format(metadata_with_efficacy_scores_df.shape[0]))
    
    metadata_with_efficacy_scores_df.to_csv(os.path.join(gv.BASE_PATH, 'full_cells_fovs/train_test_list/image_list_with_metadata__with_efficacy_scores_full_all.csv'), index=False)

    # List of columns to plot unique values for
    columns_to_plot = ['Workflow']

    # Calculate the number of unique values for each column
    unique_values_counts = {column: metadata_with_efficacy_scores_df[column].nunique() for column in columns_to_plot}

    # Plot unique values for each column
    plt.figure(figsize=(12, 8))
    plt.bar(unique_values_counts.keys(), unique_values_counts.values())
    plt.xlabel('Column Name')
    plt.ylabel('Number of Unique Values')
    plt.title('Number of Unique Values for Each Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(gv.CWD, 'figures/unique_values_plot.png'))
    plt.show()

    # Print unique values for each column
    for column in columns_to_plot:
        unique_values = metadata_with_efficacy_scores_df[column].unique()
        print(f"\nUnique values in {column}:")
        print(unique_values)

# List of columns with the results to plot
params = [
    {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5},
    {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5},
    {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5},
    {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0},
    {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5},
    {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0},
    {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5},
    {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5},
    {"organelle":"DNA","model":"../mg_model_dna_13_05_24_1.5b","noise":1.5},
    {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5},
]

# Function to plot box plots in a dynamic grid layout based on the number of params
def plot_box_plots(data, columns, params):
    colors = ['lightblue', 'green', 'blue', 'lightyellow', 'lightgray', 'lightyellow', 'lightgoldenrodyellow', 
              'lightsteelblue', 'lavender', 'honeydew', 'gray', 'green', 'yellow', 'orange', 'pink', 'brown']
    
    for x_col in tqdm(columns):
        num_plots = len(params)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)  # Dynamically calculate rows based on 3 columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24,num_rows*6))  # Adjust figure size based on rows
        fig.suptitle(f'Mask efficacy by {x_col}', fontsize=figure_config["title"], fontname=figure_config["font"])
        plt.subplots_adjust(hspace=0.3, wspace=0.1,top=0.8)
        for idx, param in enumerate(params):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case
            
            if param["organelle"] == "DNA":
                control_organelle = "Nucleolus-(Granular-Component)"
            else:
                control_organelle = param["organelle"]
                
            train_test_df = pd.read_csv(os.path.join(gv.BASE_PATH, f'full_cells_fovs/train_test_list/{control_organelle}/image_list_with_metadata__with_efficacy_scores_full.csv'))
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
            groups = [*groups,control_values]
            labels = [f'{val}' for i, val in enumerate(unique_values)]
            labels.append(f'Test set')
            
            box = ax.boxplot(groups, labels=labels, patch_artist=True, vert=True, showfliers=False)
            
            # Color the box plots
            for patch, color in zip(box['boxes'], colors[:len(groups)]):
                patch.set_facecolor(color)
            
            ax.set_ylim(0.75, max(np.max(control_values), max([group.max() for group in groups])) + 0.01)
            ax.set_title(f"{param['organelle']}",fontsize=figure_config["organelle"], fontname=figure_config["font"])
            # ax.set_xlabel(x_col,fontsize=figure_config["text"], fontname=figure_config["font"])
            if col == 0:
                ax.set_ylabel('mask efficacy [PCC]',fontsize=figure_config["axis"], fontname=figure_config["font"])
            else:
                # Turn off y-axis
                ax.get_yaxis().set_visible(False)
            
            ax.tick_params(axis='y', rotation=90)
            ax.tick_params(axis='x', rotation=0)

            # # Annotate fold change and p-values
            # for i, (fold_change, p_value) in enumerate(zip(fold_changes, p_values)):
            #     text_y = max(np.max(control_values), max([group.max() for group in groups])) + 0.05
            #     if p_value < 0.01 and np.median(groups[i]) < np.median(control_values):
            #         ax.text(i + 1,text_y+0.05, '**', ha='center', va='top', color='blue', fontsize=14, fontname=figure_config["font"])
            #     label = f"FC={fold_change:.2f}\np-value={p_value:.2g}"
            #     ax.text(i + 1,text_y+0.02, label, ha='center', va='top', fontsize=figure_config["text"], fontname=figure_config["font"])
        
        # Remove any empty subplots
        for idx in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])
        
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(gv.CWD, f'figures/{x_col}_comparison_main.png'), bbox_inches='tight',pad_inches=0.05)
        plt.close()

# Function to plot nested box plots for two categorical variables
def plot_nested_boxplots(data, col1, col2, params):
    """Plot grouped box plots for nested categorical variables"""
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Generate enough colors
    
    for param in tqdm(params, desc="Processing organelles"):
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Get unique values for both columns, sorted
        col1_values = sorted([x for x in data[col1].unique() if pd.notna(x)])
        col2_values = sorted([x for x in data[col2].unique() if pd.notna(x)])
        
        # Filter data for this organelle's model
        model_data = data[data[param["model"]].notna()]
        
        if len(model_data) == 0:
            print(f"No data found for {param['organelle']} model {param['model']}")
            continue
        
        # Create positions for boxes and collect data
        positions = []
        labels = []
        box_data = []
        box_colors = []
        
        x_pos = 0
        group_positions = []
        group_labels = []
        
        # Sort by col2 values first, then by col1 within each col2 group
        for j, val2 in enumerate(col2_values):
            group_start_pos = x_pos
            n_subgroups = 0
            
            # For each col2 value, iterate through all col1 values
            for i, val1 in enumerate(col1_values):
                subset1 = model_data[model_data[col1] == val1]
                subset2 = subset1[subset1[col2] == val2]
                
                if len(subset2) > 0:
                    efficacy_data = subset2[param["model"]].dropna()
                    if len(efficacy_data) > 0:
                        box_data.append(efficacy_data)
                        positions.append(x_pos)
                        labels.append(f'{val1}')
                        # Assign color based on col2 value
                        box_colors.append(colors[j % len(colors)])
                        x_pos += 1
                        n_subgroups += 1
            
            if n_subgroups > 0:
                group_center = group_start_pos + (n_subgroups - 1) / 2
                group_positions.append(group_center)
                group_labels.append(str(val2))
                x_pos += 0.8  # Add space between groups
        
        if len(box_data) == 0:
            print(f"No valid data for plotting {param['organelle']}")
            plt.close(fig)
            continue
        
        # Create box plot
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, 
                       widths=0.6, showfliers=False)
        
        # Color boxes by col2 values
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Formatting
        ax.set_xlabel(f'{col2}', fontsize=figure_config["axis"], fontname=figure_config["font"])
        ax.set_ylabel('Mask Efficacy [PCC]', fontsize=figure_config["axis"], fontname=figure_config["font"])
        ax.set_title(f'{param["organelle"]}', 
                    fontsize=figure_config["title"], fontname=figure_config["font"])
        
        # Set x-axis labels for groups (now col2 groups)
        if group_positions:
            ax.set_xticks(group_positions)
            ax.set_xticklabels(group_labels, fontsize=figure_config["text"], fontname=figure_config["font"])
        
        # Add col1 labels just above the x-axis
        y_min = ax.get_ylim()[0]
        label_y = y_min + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02  # 2% above the bottom
        
        for i, (pos, label) in enumerate(zip(positions, labels)):
            ax.text(pos, label_y, label, ha='center', va='bottom', 
                   fontsize=figure_config["text"], fontname=figure_config["font"],
                   rotation=0, alpha=1.0)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(gv.CWD, f'figures/{param["organelle"]}_nested_{col1}_{col2}.png'), 
                    dpi=300)
        plt.close()

# Plot box plots for each specified column and result column
# plot_box_plots(metadata_with_efficacy_scores_df, columns_to_plot, params)

# Example usage of nested plotting functions
# Uncomment and modify the column names below based on your actual data columns
print("\nAvailable columns in dataframe:")
print(metadata_with_efficacy_scores_df.columns.tolist())

# Example: Plot nested analysis for two columns
# Replace 'CellLine' and 'Workflow' with actual column names from your dataframe
col1 = 'CellLine'  # Replace with your first nested column
col2 = 'Workflow'  # Replace with your second nested column

print(f"\nUnique values in {col1}:")
print(sorted(metadata_with_efficacy_scores_df[col1].unique()))
print(f"\nUnique values in {col2}:")
print(sorted(metadata_with_efficacy_scores_df[col2].unique()))

# Create nested plots
print(f"\nCreating nested box plots for {col1} and {col2}...")
plot_nested_boxplots(metadata_with_efficacy_scores_df, col1, col2, params)
