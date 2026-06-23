import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.neighbors import KernelDensity, NearestNeighbors
import warnings
import numbers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from figure_config import figure_config
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
import pandas as pd
from matplotlib.lines import Line2D
def plot_scatter_with_shape_col(x, y, shape_col, title="Scatter plot colored by shape_col"):
    """Plot a scatter plot of x and y, colored by shape_col and axes limited by 2nd and 98th percentiles."""
    
    # Ensure x, y, and shape_col are numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    shape_col = np.asarray(shape_col).flatten()

    # Create a mapping of unique values in shape_col to colors
    unique_shapes = np.unique(shape_col)
    color_map = mcolors.ListedColormap(plt.cm.tab10.colors[:len(unique_shapes)])  # Use the tab10 colormap for a range of colors
    
    # Map each value in shape_col to a color
    shape_color_map = {shape: color_map(i) for i, shape in enumerate(unique_shapes)}
    point_colors = np.array([shape_color_map[shape] for shape in shape_col])

    # Calculate the 2nd and 98th percentiles for x and y
    x_2nd_percentile = np.percentile(x, 1)
    x_98th_percentile = np.percentile(x, 100)
    y_2nd_percentile = np.percentile(y, 1)
    y_98th_percentile = np.percentile(y, 100)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=point_colors, s=50, marker='o', edgecolor='k')

    # Set axis limits based on the 2nd and 98th percentiles
    plt.xlim(x_2nd_percentile, x_98th_percentile)
    plt.ylim(y_2nd_percentile, y_98th_percentile)

    # Create legend from unique values in shape_col
    legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i), markersize=10, label=shape) 
                     for i, shape in enumerate(unique_shapes)]
    plt.legend(handles=legend_labels, title="Shape Categories", loc="upper left")

    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(os.path.join( gv.CWD, "figures/{}.png".format(title)))

def dermi_kde(x, y, x_col="Feature1",y_col="Feature2", n_bins=20, n_mesh=3, plot=True,title=""):
    """Compute Density Resampled Estimate of Mutual Information using KDE."""
    x, y = _vector_coerce_two_dense(x, y)

    if np.count_nonzero(x - x[0]) == 0 or np.count_nonzero(y - y[0]) == 0:
        warnings.warn(
            "Attempting to calculate DREMI on a constant array. Returning `0`",
            UserWarning,
        )
        return 0

    # Z-score X and Y
    # x = np.clip(x,-5,5) #np.clip(stats.zscore(x),-5,5)
    # y = np.clip(y,-5,5) #np.clip(stats.zscore(y),-5,5)

    # Create bin and mesh points
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), ((n_mesh + 1) * n_bins) + 1)
    y_mesh = np.linspace(min(y), max(y), ((n_mesh + 1) * n_bins) + 1)

    # Perform joint density estimation using KDE
    data_subset = np.vstack([x, y])
    kde_joint = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_subset.T)

    # Evaluate joint density on the mesh points
    mesh_points = np.vstack(
        [np.tile(x_mesh, len(y_mesh)), np.repeat(y_mesh, len(x_mesh))]
    ).T
    log_joint_density = kde_joint.score_samples(mesh_points)
    joint_density = np.exp(log_joint_density)

    # Sum the densities of each point over the bins
    bin_density, _, _ = np.histogram2d(
        mesh_points[:, 0],
        mesh_points[:, 1],
        bins=[x_bins, y_bins],
        weights=joint_density,
    )
    bin_density = bin_density.T
    bin_density = bin_density / np.sum(bin_density)

    # Compute conditional density estimate of Y given X
    conditional_density = bin_density / np.sum(bin_density, axis=0)
    
    # Rescale each column by its maximum value
    column_max = np.max(conditional_density, axis=0)
    rescaled_density = conditional_density / column_max

    # Mutual information
    marginal_entropy = stats.entropy(np.sum(bin_density, axis=1), base=2)
    cond_entropies = stats.entropy(conditional_density, base=2)
    cond_sums = np.sum(bin_density, axis=0)
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy

    # DREMI
    marginal_entropy_norm = stats.entropy(np.sum(conditional_density, axis=1), base=2)
    cond_sums_norm = np.mean(conditional_density)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)
    dremi = marginal_entropy_norm - conditional_entropy_norm

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        
        # Calculate the difference between y and x
        diff = y - x  # Difference between y[i] and x[i]

        # Clip the difference values between -5 and 5
        clipped_diff = np.clip(diff, -5, 5)

        # Create a mask for points where the difference is between -2 and 2
        grey_mask = (diff >= -2) & (diff <= 2)

        # Create a custom colormap with blue for low values, grey for -2 to 2, and red for high values
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_colormap", 
            ["blue", "grey", "red"], 
            N=256
        )

        # Set up a Normalize instance to fix the colorbar range from -5 to 5
        norm = Normalize(vmin=-5, vmax=5)

        # Plot grey points first, where diff is between -2 and 2
        axes[0].scatter(x[grey_mask], y[grey_mask], c='grey', s=4, label='Difference between -2 and 2')

        # Plot the remaining points using the custom colormap with the fixed norm
        scatter = axes[0].scatter(x[~grey_mask], y[~grey_mask], c=clipped_diff[~grey_mask], cmap=cmap, norm=norm, s=4, label='Other differences')

        # Add a colorbar with the custom colormap and the fixed range
        cbar = fig.colorbar(scatter, ax=axes[0], label='Clipped Difference (y - x)', ticks=[-5, -2, 0, 2, 5])
        cbar.set_ticklabels(['-5', '-2', '0', '2', '5'])
        axes[0].set_title("Raw scores", fontsize=14, fontname=figure_config["font"])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])
        axes[0].set_ylabel(y_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot KDE joint density
        n = ((n_mesh + 1) * n_bins) + 1
        axes[1].imshow(
            joint_density.reshape(n, n), cmap="inferno", origin="lower", aspect="auto"
        )
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axhline(b - 0.5, c="grey", linewidth=1)
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axvline(b - 0.5, c="grey", linewidth=1)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title("KDE Density", fontsize=14, fontname=figure_config["font"])
        axes[1].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot joint probability
        axes[2].imshow(bin_density, cmap="inferno", origin="lower", aspect="auto")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_title("Joint Prob.[MI={:.2f}]".format(mutual_info), fontsize=14, fontname=figure_config["font"])
        axes[2].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot conditional probability
        axes[3].imshow(rescaled_density, cmap="inferno", origin="lower", aspect="auto")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_title("Conditional Prob.[DREMI={:.2f}]".format(dremi), fontsize=14, fontname=figure_config["font"])
        axes[3].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        fig.suptitle(title, fontsize=figure_config["organelle"],fontname=figure_config["font"],y=0.92)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join( gv.CWD, "figures/kde_dermi_{}.png".format(title)))

    return dremi

def dermi_knn(x, y, x_col="Feature1",y_col="Feature2",k=5, n_bins=20, n_mesh=3, n_jobs=1, plot=False,title=""):
    """Compute kNN conditional Density Resampled Estimate of Mutual Information."""
    x, y = _vector_coerce_two_dense(x, y)

    if np.count_nonzero(x - x[0]) == 0 or np.count_nonzero(y - y[0]) == 0:
        warnings.warn(
            "Attempting to calculate kNN-DREMI on a constant array. Returning `0`",
            UserWarning,
        )
        return -1

    if not isinstance(k, numbers.Integral):
        raise ValueError("Expected k as an integer. Got {}".format(type(k)))
    if not isinstance(n_bins, numbers.Integral):
        raise ValueError("Expected n_bins as an integer. Got {}".format(type(n_bins)))
    if not isinstance(n_mesh, numbers.Integral):
        raise ValueError("Expected n_mesh as an integer. Got {}".format(type(n_mesh)))

    # Z-score X and Y
    # x = stats.zscore(x) #np.clip(stats.zscore(x),-5,5)
    # y = stats.zscore(y) #np.clip(stats.zscore(y),-5,5)
    # x = np.clip(x,-5,5) #np.clip(stats.zscore(x),-5,5)
    # y = np.clip(y,-5,5) #np.clip(stats.zscore(y),-5,5)
    
    # Create bin and mesh points
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), ((n_mesh + 1) * n_bins) + 1)
    y_mesh = np.linspace(min(y), max(y), ((n_mesh + 1) * n_bins) + 1)

    mesh_points = np.vstack(
        [np.tile(x_mesh, len(y_mesh)), np.repeat(y_mesh, len(x_mesh))]
    ).T

    knn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(np.vstack([x, y]).T)
    dists, _ = knn.kneighbors(mesh_points)

    area = np.pi * (dists[:, -1] ** 2)
    density = k / (area + 1e-10)  # Add small constant to avoid division by zero

    mesh_mask = np.logical_or(
        np.isin(mesh_points[:, 0], x_bins), np.isin(mesh_points[:, 1], y_bins)
    )
    bin_density, _, _ = np.histogram2d(
        mesh_points[~mesh_mask, 0],
        mesh_points[~mesh_mask, 1],
        bins=[x_bins, y_bins],
        weights=density[~mesh_mask],
    )
    bin_density = bin_density.T
    bin_density = bin_density / np.sum(bin_density)

    drevi = bin_density / np.sum(bin_density, axis=0)
    cond_entropies = stats.entropy(drevi, base=2)

    marginal_entropy = stats.entropy(np.sum(bin_density, axis=1), base=2)
    cond_sums = np.sum(bin_density, axis=0)
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy

    marginal_entropy_norm = stats.entropy(np.sum(drevi, axis=1), base=2)
    cond_sums_norm = np.mean(drevi)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)

    dremi = marginal_entropy_norm - conditional_entropy_norm

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        
        # Plot raw data
        axes[0].scatter(x, y, c="k", s=4)
        axes[0].set_title("Row scores",fontsize=14, fontname=figure_config["font"])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])
        axes[0].set_ylabel(y_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot kNN density
        n = ((n_mesh + 1) * n_bins) + 1
        axes[1].imshow(
            np.log(density.reshape(n, n)), cmap="inferno", origin="lower", aspect="auto"
        )
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axhline(b - 0.5, c="grey", linewidth=1)

        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axvline(b - 0.5, c="grey", linewidth=1)

        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title("kNN Density", fontsize=14, fontname=figure_config["font"])
        axes[1].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot joint probability
        axes[2].imshow(bin_density, cmap="inferno", origin="lower", aspect="auto")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_title(
            "Joint Prob.[MI={:.2f}]".format(mutual_info), fontsize=14, fontname=figure_config["font"]
        )
        axes[2].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot conditional probability
        axes[3].imshow(drevi, cmap="inferno", origin="lower", aspect="auto")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_title(
            "Conditional Prob.[DREMI={:.2f}]".format(dremi), fontsize=14, fontname=figure_config["font"]
        )
        axes[3].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        fig.suptitle(title, fontsize=figure_config["organelle"], fontname=figure_config["font"],y=0.92)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join( gv.CWD, "figures/knn_dermi_{}.png".format(title)))

        return dremi

def _vector_coerce_two_dense(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) != len(y):
        raise ValueError("Expected x and y to be the same length. Got {} and {}".format(len(x), len(y)))
    return x, y

# import pandas as pd
import global_vars as gv

# corr_df = pd.read_csv(os.path.join(gv.DATA_PATH, "unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv"))
# x_col = '../mg_model_membrane_13_05_24_1.5'
# y_col = '../unet_model_22_05_22_membrane_128'

# corr_df_temp = corr_df.dropna(how='any',inplace=False, subset=[x_col,y_col])

# x = corr_df_temp[x_col].values.reshape((-1,1))
# y = corr_df_temp[y_col].values.reshape((-1,1))

# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# x = scaler_x.fit(corr_df_temp[corr_df_temp['StructureDisplayName']=="Plasma membrane"][x_col].values.reshape((-1,1))).transform(x)
# y = scaler_y.fit(corr_df_temp[corr_df_temp['StructureDisplayName']=="Plasma membrane"][y_col].values.reshape((-1,1))).transform(y)
# title = "Plasma-membrane"
# # Plot DREMI with KDE and kNN-DREMI
# print("Plotting DREMI with KDE:")
# dremi_value_kde = dermi_kde(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", n_bins=20, n_mesh=20, plot=True,title=title)

# print("Plotting kNN-DREMI:")
# dremi_value_knn = dermi_knn(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", k=7, n_bins=20, n_mesh=20, n_jobs=2, plot=True,title=title)

# print("DREMI with KDE:", dremi_value_kde)
# print("kNN-DREMI:", dremi_value_knn)
# plot_scatter_with_shape_col(x,y,corr_df_temp['Workflow'].values,"scatter_{}".format("Plasma-membrane"))

# # corr_df = pd.read_csv(os.path.join(gv.CWD, "figures/dna_correlation.csv"))
# x_col = '../mg_model_dna_13_05_24_1.5b'
# y_col = '../unet_model_22_05_22_dna_128'

# corr_df_temp = corr_df.dropna(how='any',inplace=False, subset=[x_col,y_col])
# x = corr_df_temp[x_col].values.reshape((-1,1))
# y = corr_df_temp[y_col].values.reshape((-1,1))

# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# x = scaler_x.fit(corr_df_temp[corr_df_temp['StructureDisplayName']=="Nucleolus (Granular Component)"][x_col].values.reshape((-1,1))).transform(x)
# y = scaler_y.fit(corr_df_temp[corr_df_temp['StructureDisplayName']=="Nucleolus (Granular Component)"][y_col].values.reshape((-1,1))).transform(y)

# title = "DNA"
# # Plot DREMI with KDE and kNN-DREMI
# print("Plotting DREMI with KDE:")
# dremi_value_kde = dermi_kde(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", n_bins=20, n_mesh=20, plot=True,title=title)

# print("Plotting kNN-DREMI:")
# dremi_value_knn = dermi_knn(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", k=7, n_bins=20, n_mesh=20, n_jobs=2, plot=True,title=title)

# print("DREMI with KDE:", dremi_value_kde)
# print("kNN-DREMI:", dremi_value_knn)
# plot_scatter_with_shape_col(x,y,corr_df_temp['Workflow'].values,"scatter_{}".format("DNA"))

##################################
# Perturbations data

params = [
    
    # {"organelle":"Golgi", "y_col":"../unet_model_22_05_22_golgi_128", "x_col":"../mg_model_golgi_13_05_24_1.5","drop":None},
    # {"organelle":"Microtubules", "y_col":"../unet_model_22_05_22_microtubules_128", "x_col":"../mg_model_microtubules_13_05_24_1.5","drop":None},
    {"organelle":"Plasma-membrane", "y_col":"../unet_model_22_05_22_membrane_128", "x_col":"../mg_model_membrane_13_05_24_1.5","drop":[0,2,6]},
    # {"organelle":"DNA", "y_col":"../unet_model_22_05_22_dna_128", "x_col":"../mg_model_dna_13_05_24_1.5b","drop":None},
    # {"organelle":"Actin filaments", "y_col":"../unet_model_22_05_22_actin_128", "x_col":"../mg_model_actin_13_05_24_1.5","drop":None},
    # {"organelle":"Actomyosin bundles", "y_col":"../unet_model_22_05_22_bundles_128", "x_col":"../mg_model_bundles_13_05_24_1.0","drop":None},
    # {"organelle":"Endoplasmic reticulum", "y_col":"../unet_model_22_05_22_er_128", "x_col":"../mg_model_er_13_05_24_1.5","drop":[1]},
]
corr_df = pd.read_csv(os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv'))
drug_label = ['s-Nitro-Blebbistatin']#,'Staurosporine'] #,'Staurosporine'
# corr_df = corr_df[ ~(corr_df['drug_label'].isin(drug_label))]

for param in params:
    print(param["organelle"])
    corr_df = pd.read_csv(os.path.join(gv.BASE_PATH, 'full_cells_fovs_perturbation/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores_embeddings_{}.csv'.format(param["organelle"])))
    
    x_col = param["x_col"]
    y_col = param["y_col"]
    corr_df_temp = corr_df.dropna(how='any',inplace=False, subset=[x_col,y_col])
    if param["drop"] is not None:
        corr_df_temp = corr_df_temp[~(corr_df_temp['cluster'].isin(param["drop"]))]
    # corr_df_temp = corr_df_temp[(corr_df_temp[x_col]>corr_df_temp[x_col].quantile(0.00)) & (corr_df_temp[x_col]>corr_df_temp[y_col].quantile(0.00))]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    x = corr_df_temp[x_col].values.reshape((-1,1))
    y = corr_df_temp[y_col].values.reshape((-1,1))
    #corr_df_temp[corr_df_temp['drug_label']=="Vehicle"][x_col].values.reshape((-1,1))
    #corr_df_temp[corr_df_temp['drug_label']=="Vehicle"][y_col].values.reshape((-1,1))
    x = scaler_x.fit(x).transform(x)
    y = scaler_y.fit(y).transform(y)
    
    
    title = "{}-perturbations-except-{}".format(param["organelle"],drug_label)

    # Plot DREMI with KDE and kNN-DREMI
    print("Plotting DREMI with KDE:")
    dremi_value_kde = dermi_kde(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", n_bins=20, n_mesh=20, plot=True,title=title)

    print("Plotting kNN-DREMI:")
    dremi_value_knn = dermi_knn(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", k=5, n_bins=20, n_mesh=20, n_jobs=2, plot=True,title=title)

    print("DREMI with KDE:", dremi_value_kde)
    print("kNN-DREMI:", dremi_value_knn)
    
    plot_scatter_with_shape_col(x,y,corr_df_temp['cluster'].values,"scatter_pertrub_{}".format(param["organelle"]))
    
    
    