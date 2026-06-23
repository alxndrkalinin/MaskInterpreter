# Quick Start with Example Notebook

The [example.ipynb](example.ipynb) notebook provides a complete walkthrough of loading a pre-trained MaskInterpreter model and analyzing explanation masks. It demonstrates:

1. **Environment setup** - Setting required paths
2. **Loading data** - Using the DataGen class to load test images
3. **Loading models** - Loading pre-trained predictor and MaskInterpreter models
4. **Generating masks** - Creating importance masks for test images
5. **Mask analysis** - Calculating mask efficacy (PCC) and mask size metrics
6. **Visualization** - Plotting original images, predictions, masks, and noisy predictions

## Running the Example

Before running the notebook, ensure you have:
- Configured paths in `global_vars.py` or in the notebook
- Downloaded the example data or have your own dataset prepared
- Pre-trained models available in the configured models directory

Open the notebook:
```bash
jupyter notebook example.ipynb
```

Or use VS Code's built-in Jupyter support to run the cells interactively.

The notebook will guide you through the complete analysis pipeline and generate visualizations showing how MaskInterpreter identifies important regions in your images.
