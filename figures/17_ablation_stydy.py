import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from figure_config import figure_config
# Path to CSV files
csv_path = os.path.join(gv.BASE_PATH, 'loss graphs/')
csv_files = glob.glob(f"{csv_path}*.csv")
csv_files = sorted(csv_files)

print(f"Found {len(csv_files)} CSV files")

# Load all CSV files and extract loss weights from filenames
data_list = []
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    print(f"Loading: {filename}")
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    
    # Extract loss weights from filename using regex
    match = re.search(r'sim[_=]?(\d+\.?\d*).*?target[_=]?(\d+\.?\d*).*?mask[_=]?(\d+\.?\d*)', filename, re.IGNORECASE)
    
    if match:
        sim_weight = float(match.group(1))
        target_weight = float(match.group(2))
        mask_weight = float(match.group(3))
        label = f"S{sim_weight}\nT{target_weight}\nM{mask_weight}"
    else:
        sim_weight = None
        target_weight = None
        mask_weight = None
        label = filename.replace('.csv', '')
    
    # Identify which term is zero
    zero_term = None
    if sim_weight == 0:
        zero_term = 'sim'
    elif target_weight == 0:
        zero_term = 'target'
    elif mask_weight == 0:
        zero_term = 'mask'
    else:
        zero_term = 'none'
    
    data_list.append({
        'filename': filename,
        'label': label,
        'data': df,
        'csv_file': csv_file,
        'sim_weight': sim_weight,
        'target_weight': target_weight,
        'mask_weight': mask_weight,
        'zero_term': zero_term
    })

# ============================================================================
# Extract metrics at best epoch (min val_total_loss)
# ============================================================================
metrics_summary = []

for data_info in data_list:
    df = data_info['data']
    label = data_info['label']
    
    if 'val_total_loss' in df.columns:
        # Find epoch with minimum val_total_loss
        min_idx = df['val_total_loss'].idxmin()
        best_epoch = df.loc[min_idx, 'epoch'] if 'epoch' in df.columns else min_idx
        
        # Extract metrics at this epoch
        metrics = {
            'label': label,
            'best_epoch': best_epoch,
            'val_total_loss': df.loc[min_idx, 'val_total_loss'],
            'zero_term': data_info['zero_term'],
            'sim_weight': data_info['sim_weight'],
            'target_weight': data_info['target_weight'],
            'mask_weight': data_info['mask_weight']
        }
        
        # Add val_pcc if available
        if 'val_pcc' in df.columns:
            metrics['val_pcc'] = df.loc[min_idx, 'val_pcc']
        
        # Add val_similarity_loss (check both spellings)
        if 'val_similiarity_loss' in df.columns:
            metrics['val_similarity_loss'] = df.loc[min_idx, 'val_similiarity_loss']
        elif 'val_similarity_loss' in df.columns:
            metrics['val_similarity_loss'] = df.loc[min_idx, 'val_similarity_loss']
        
        # Add val_importance_mask_size if available
        if 'val_importance_mask_size' in df.columns:
            metrics['val_importance_mask_size'] = df.loc[min_idx, 'val_importance_mask_size']
        
        metrics_summary.append(metrics)
        print(f"\n{label.replace(chr(10), ' ')}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Val total loss: {metrics['val_total_loss']:.4f}")
        if 'val_pcc' in metrics:
            print(f"  Val PCC: {metrics['val_pcc']:.4f}")
        if 'val_similarity_loss' in metrics:
            print(f"  Val similarity loss: {metrics['val_similarity_loss']:.4f}")
        if 'val_importance_mask_size' in metrics:
            print(f"  Val importance mask size: {metrics['val_importance_mask_size']:.4f}")

# ============================================================================
# Combined Ablation Study Figure - Nature A4 width (180mm = 7.08 inches)
# ============================================================================
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpecFromSubplotSpec
import global_vars as gv

if metrics_summary:
    # Find all required configurations
    config_without_sim = None  # S0.0 T6.0 M1.0
    config_with_sim = None     # S1.0 T6.0 M1.0
    config_without_mask = None  # S1.0 T6.0 M0.0
    config_with_mask = None     # S1.0 T6.0 M1.0
    
    for m in metrics_summary:
        if m['sim_weight'] == 0.0 and m['target_weight'] == 6.0 and m['mask_weight'] == 1.0:
            config_without_sim = m
        if m['sim_weight'] == 1.0 and m['target_weight'] == 6.0 and m['mask_weight'] == 1.0:
            config_with_sim = m
            config_with_mask = m
        if m['sim_weight'] == 1.0 and m['target_weight'] == 6.0 and m['mask_weight'] == 0.0:
            config_without_mask = m
    
    # Find configurations for target comparison
    configs_t0 = [m for m in metrics_summary if m['sim_weight'] == 1.0 and m['target_weight'] == 0.0 and m['mask_weight'] != 0.0]
    configs_t6 = [m for m in metrics_summary if m['sim_weight'] == 1.0 and m['target_weight'] == 6.0 and m['mask_weight'] != 0.0]
    
    # Check if params_comparison.png exists
    params_img_path = os.path.join(gv.CWD, 'figures/params_comparison.png')
    has_params_img = os.path.exists(params_img_path)
    
    # New layout:
    # Row 0 (height 1): A - Target (spans all 3 columns)
    # Row 1-2 (height 2): B (col 0, row 1) + C (col 0, row 2) + D (cols 1-2, rows 1-2)
    fig = plt.figure(figsize=(7.08, 8))
    # Main grid: 2 rows with large spacing between row 0 and row 1
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.6], hspace=0.2)
    # Top row for A
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.15)
    # Bottom section for B, C, D with smaller internal spacing
    gs_bottom = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[1], height_ratios=[1, 1], 
                                         width_ratios=[1.2, 0.85, 0.85], hspace=0.15, wspace=0.15)
    
    # Nature-style font sizes (smaller for publication)
    fontsize_label = 8
    fontsize_tick = 7
    fontsize_legend = 6
    fontsize_values = 6
    fontsize_ylabel = 6
    
    metrics_to_plot_simple = [
        ('val_pcc', 'PCC'),
        ('val_similarity_loss', 'Similarity Loss'),
        ('val_importance_mask_size', 'Mask Size')
    ]
    
    width = 0.35
    
    # =========================================================================
    # Row 0: Target comparison (3 subplots) - Panel A (spans all columns)
    # =========================================================================
    if configs_t0 or configs_t6:
        axes_target = [fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1]), fig.add_subplot(gs_top[0, 2])]
        
        metrics_to_plot_target = [
            ('val_pcc', 'PCC', axes_target[0]),
            ('val_similarity_loss', 'Similarity Loss', axes_target[1]),
            ('val_importance_mask_size', 'Mask Size', axes_target[2])
        ]
        
        for idx, (metric_key, metric_title, ax) in enumerate(metrics_to_plot_target):
            all_configs = []
            
            for m in configs_t0:
                if metric_key in m:
                    label = f"M={m['mask_weight']}"
                    all_configs.append((label, m[metric_key], '#1f77b4', 0, m['mask_weight']))
            
            for m in configs_t6:
                if metric_key in m:
                    label = f"M={m['mask_weight']}"
                    all_configs.append((label, m[metric_key], '#ff7f0e', 6, m['mask_weight']))
            
            if not all_configs:
                continue
            
            # Sort by T first, then by M ascending
            all_configs = sorted(all_configs, key=lambda x: (x[3], x[4]))
            
            labels = [c[0] for c in all_configs]
            values = [c[1] for c in all_configs]
            colors = [c[2] for c in all_configs]
            target_vals = [c[3] for c in all_configs]
            
            bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.8)
            
            prev_t = None
            for i, t in enumerate(target_vals):
                if prev_t is not None and t != prev_t:
                    ax.axhline(y=i - 0.5, color='black', linewidth=1, linestyle='-')
                prev_t = t
            
            ax.set_yticks(range(len(labels)))
            if idx == 0:
                ax.set_yticklabels(labels, fontsize=fontsize_tick)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(metric_title, fontsize=fontsize_label)
            ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
            ax.tick_params(labelsize=fontsize_tick)
            
            x_max = max(values) if values else 1
            ax.set_xlim([0, x_max * 1.25])
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=fontsize_values, fontweight='bold')
            
            if idx == 0:
                ax.text(-0.25, 1.05, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
        
        # Add legend for target comparison at the top
        legend_elements_target = [
            Patch(facecolor='#1f77b4', edgecolor='black', label='Without Target (S=1, T=0, M=*)'),
            Patch(facecolor='#ff7f0e', edgecolor='black', label='With Target (S=1, T=6, M=*)')
        ]
        axes_target[1].legend(handles=legend_elements_target, fontsize=fontsize_legend, 
                              loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, framealpha=0.9)
    
    # =========================================================================
    # Row 1, Col 0: Similarity comparison - Panel B
    # =========================================================================
    if config_without_sim and config_with_sim:
        ax_sim = fig.add_subplot(gs_bottom[0, 0])
        
        without_sim_vals = []
        with_sim_vals = []
        metric_labels = []
        
        for metric_key, metric_label in metrics_to_plot_simple:
            if metric_key in config_without_sim and metric_key in config_with_sim:
                without_sim_vals.append(config_without_sim[metric_key])
                with_sim_vals.append(config_with_sim[metric_key])
                metric_labels.append(metric_label)
        
        x = np.arange(len(metric_labels))
        
        bars1 = ax_sim.bar(x - width/2, without_sim_vals, width, label='Without Similarity (S=0, T=6, M=1)', 
                           color='#ff7f0e', edgecolor='black', linewidth=0.8)
        bars2 = ax_sim.bar(x + width/2, with_sim_vals, width, label='With Similarity (S=1, T=6, M=1)', 
                           color='#1f77b4', edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars1, without_sim_vals):
            ax_sim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize_values, fontweight='bold')
        for bar, val in zip(bars2, with_sim_vals):
            ax_sim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize_values, fontweight='bold')
        
        ax_sim.set_ylabel('Value', fontsize=fontsize_ylabel)
        ax_sim.set_xticks(x)
        ax_sim.set_xticklabels(metric_labels, fontsize=fontsize_tick)
        ax_sim.legend(fontsize=fontsize_legend-1, loc='upper left', bbox_to_anchor=(0.0, 0.99), ncol=1, framealpha=0.9)
        ax_sim.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax_sim.tick_params(labelsize=fontsize_tick)
        ax_sim.set_ylim([0, max(max(without_sim_vals), max(with_sim_vals)) * 1.35])
        ax_sim.text(-0.15, 1.05, 'B', transform=ax_sim.transAxes, fontsize=10, fontweight='bold', va='top')
    
    # =========================================================================
    # Row 2, Col 0: Mask comparison - Panel C
    # =========================================================================
    if config_without_mask and config_with_mask:
        ax_mask = fig.add_subplot(gs_bottom[1, 0])
        
        without_mask_vals = []
        with_mask_vals = []
        metric_labels = []
        
        for metric_key, metric_label in metrics_to_plot_simple:
            if metric_key in config_without_mask and metric_key in config_with_mask:
                without_mask_vals.append(config_without_mask[metric_key])
                with_mask_vals.append(config_with_mask[metric_key])
                metric_labels.append(metric_label)
        
        x = np.arange(len(metric_labels))
        
        bars1 = ax_mask.bar(x - width/2, without_mask_vals, width, label='Without Mask (S=1, T=6, M=0)', 
                            color='#1f77b4', edgecolor='black', linewidth=0.8)
        bars2 = ax_mask.bar(x + width/2, with_mask_vals, width, label='With Mask (S=1, T=6, M=1)', 
                            color='#ff7f0e', edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars1, without_mask_vals):
            ax_mask.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize_values, fontweight='bold')
        for bar, val in zip(bars2, with_mask_vals):
            ax_mask.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize_values, fontweight='bold')
        
        ax_mask.set_ylabel('Value', fontsize=fontsize_ylabel)
        ax_mask.set_xticks(x)
        ax_mask.set_xticklabels(metric_labels, fontsize=fontsize_tick)
        ax_mask.legend(fontsize=fontsize_legend-1, loc='upper left', bbox_to_anchor=(0.0, 0.99), ncol=1, framealpha=0.9)
        ax_mask.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax_mask.tick_params(labelsize=fontsize_tick)
        ax_mask.set_ylim([0, max(max(without_mask_vals), max(with_mask_vals)) * 1.35])
        ax_mask.text(-0.15, 1.05, 'C', transform=ax_mask.transAxes, fontsize=10, fontweight='bold', va='top')
    
    # =========================================================================
    # Rows 1-2, Cols 1-2: Params comparison image - Panel D (2x2)
    # =========================================================================
    if has_params_img:
        ax_params = fig.add_subplot(gs_bottom[0:2, 1:3])
        params_img = plt.imread(params_img_path)
        ax_params.imshow(params_img, aspect='equal')
        ax_params.axis('off')
        ax_params.text(-0.03, 1.03, 'D', transform=ax_params.transAxes, fontsize=10, fontweight='bold', va='top')
    
    save_path = os.path.join(gv.CWD, 'figures/ablation_study_combined.png')
    plt.savefig(save_path,
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"\nSaved: {save_path}")
    plt.close()

print("\nAblation study plots completed!")
