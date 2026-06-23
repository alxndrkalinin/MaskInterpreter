# Recreate Figures from Paper

To reproduce all figures from the paper, run the scripts in the `figures/` folder in numerical order. 

**Important**: Before running, update the following paths in each script:
- Model checkpoint paths (trained MaskInterpreter and predictor models)
- Data directory paths (Allen Cell Collection dataset location)
- Output directory paths for generated figures

## Running Scripts in Order

```bash
cd figures

# 0. Reproduce U-Net scores
python 0_reproduce_unet_scores.py

# 1. Choose noise scale parameter
python 1_choose_noise_scale.py

# 2. Choose threshold for binarization
python 2_choose_th.py

# 3. Calculate U-Net prediction scores
python 3_calculate_unet_scores.py

# 4. Calculate explanation mask efficacy
python 4_calculate_explanation_mask_efficacy.py

...
```

**Note**: Some scripts may have dependencies on outputs from previous scripts. Run them in the order listed above to ensure all required intermediate files are generated.
