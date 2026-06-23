# Full Data Download

## Allen Cell Collection Dataset

The project uses the Allen Cell Collection dataset from AWS S3. To download and prepare the full field-of-view (FOV) dataset:

**Step 1: Download images**
```bash
cd create_data
python download_and_create_dataset_full.py
```

**Step 2: Create metadata**

After downloading the images, run the metadata creation script:
```bash
python create_metadata.py
```

This second step processes the downloaded images and generates the metadata CSV files required for training.

## Configuration Parameters

Edit the script to customize the download. Key parameters in `download_and_create_dataset_full.py`:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `num_threads` | Number of parallel download threads | `4` |
| `storage_root` | Main directory to save downloaded data | `"/path/to/your/storage"` |
| `temp_storage_root` | Temporary directory for processing (use SSD for speed) | `"/path/to/temp"` |
| `num_of_images_per_organelle` | Maximum images to download per organelle | `200` |
| `resacle_z` | Z-axis rescaling factor | `3` |
| `only_csvs` | If `True`, creates metadata CSVs only (no images) | `True` |
| `override` | If `True`, re-downloads existing images | `False` |
| `organelles` | Dictionary of organelles to download | See below |

## Available Organelles

The script supports downloading the following organelles (uncomment in the `organelles` dictionary):

```python
organelles = {
    "Golgi": [],
    "Microtubules": [],
    "Nuclear-envelope": [],
    "Actin-filaments": [],
    "Mitochondria": [],
    "Endoplasmic-reticulum": [],
    "Nucleolus-(Granular-Component)": [],
    "Actomyosin-bundles": [],
    "Plasma-membrane": [],
}
```

## Output Structure

Each downloaded image is a multi-channel TIFF with:
- **Channel 0**: Bright-field image (ROI)
- **Channel 1**: DNA fluorescence (raw ROI)
- **Channel 2**: Membrane fluorescence (raw ROI)
- **Channel 3**: Structure/organelle fluorescence (raw ROI)
- **Channel 4**: DNA segmentation (ROI)
- **Channel 5**: Membrane segmentation (ROI)
- **Channel 6**: Structure/organelle segmentation (ROI)

The script also generates a metadata CSV for each organelle with image paths and metadata.

## Example Usage

To download 50 images each of Mitochondria and Golgi:

```python
# Edit download_and_create_dataset_full.py
num_of_images_per_organelle = 50
only_csvs = False  # Actually download images
organelles = {
    "Mitochondria": [],
    "Golgi": []
}
```

Then run:
```bash
python download_and_create_dataset_full.py
```
