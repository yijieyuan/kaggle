import numpy as np
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
series_dir = Path(r'E:\data_old\series')
output_viz_dir = Path('./multi-frame_visualization')
output_viz_dir.mkdir(exist_ok=True)

# Find all series folders
all_series_folders = [f for f in series_dir.iterdir() if f.is_dir()]

print(f"Total series folders: {len(all_series_folders)}")

# Process each series
for series_folder in all_series_folders:
    dcm_files = list(series_folder.glob("*.dcm"))
    
    # Check if multi-frame (only 1 DICOM file)
    if len(dcm_files) != 1:
        continue
    
    try:
        # Read the multi-frame DICOM
        ds = pydicom.dcmread(dcm_files[0])
        volume = ds.pixel_array
        
        # Check if it's 3D
        if len(volume.shape) != 3:
            continue
        
        series_uid = ds.SeriesInstanceUID
        print(f"Processing: {series_uid}, shape: {volume.shape}")
        
        # Get middle slices
        mid_dim0 = volume.shape[0] // 2
        mid_dim1 = volume.shape[1] // 2
        mid_dim2 = volume.shape[2] // 2
        
        # Extract slices
        slice_dim0 = volume[mid_dim0, :, :]
        slice_dim1 = volume[:, mid_dim1, :]
        slice_dim2 = volume[:, :, mid_dim2]
        
        # Create 1x3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(slice_dim0, cmap='gray')
        axes[0].set_title('dim0')
        axes[0].axis('off')
        
        axes[1].imshow(slice_dim1, cmap='gray')
        axes[1].set_title('dim1')
        axes[1].axis('off')
        
        axes[2].imshow(slice_dim2, cmap='gray')
        axes[2].set_title('dim2')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_viz_dir / f'{series_uid}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error processing {series_folder.name}: {e}")
        continue

print("\nDone!")