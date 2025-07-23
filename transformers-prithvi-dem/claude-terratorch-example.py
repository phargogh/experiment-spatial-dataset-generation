import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds
from terratorch.datasets import HLSBands
from terratorch.models import PrithviModelFactory
from terratorch.tasks import SemanticSegmentationTask


# Initialize Prithvi model using terratorch
def load_prithvi_model():
    """Load Prithvi model using terratorch factory"""
    # Had to fill in these parameters based on the docstring
    # Required: 'task', 'backbone', 'decoder', and 'bands'
    model_factory = PrithviModelFactory()

    model = model_factory.build_model(
        task='regression',  # task (default)
        backbone="prithvi_eo_v2_600_tl",  # backbone (default)
        decoder="FCNDecoder",  # decoder (default)
        bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],  # bands (default)
        num_frames=1,  # Single timestamp
        pretrained=True
    )

    return model

def create_synthetic_input_data(batch_size=1, channels=6, height=224, width=224):
    """Create synthetic satellite imagery input"""
    # Generate synthetic HLS-like data (6 bands)
    # Bands: Blue, Green, Red, NIR, SWIR1, SWIR2

    # Create realistic spectral signatures
    base_data = torch.zeros(batch_size, channels, height, width)

    # Simulate different land cover types
    for i in range(height):
        for j in range(width):
            # Water (low in all bands)
            if (i < height//3) and (j < width//3):
                #base_data[:, :, i, j] = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.01, 0.01])
                base_data[:, :, i, j] = torch.tensor([0.02, 0.04, 0.06])

            # Vegetation (high NIR, moderate visible)
            elif (i < height//3) and (j >= width//3) and (j < 2*width//3):
                base_data[:, :, i, j] = torch.tensor([0.05, 0.08, 0.06])

            # Urban (moderate in all bands)
            elif (i < height//3):
                base_data[:, :, i, j] = torch.tensor([0.15, 0.18, 0.20])

            # Bare soil (increasing towards SWIR)
            elif (i < 2*height//3):
                base_data[:, :, i, j] = torch.tensor([0.12, 0.16, 0.22])

            # Mixed/other
            else:
                base_data[:, :, i, j] = torch.tensor([0.10, 0.12, 0.15])

    # Add some noise for realism
    noise = torch.randn_like(base_data) * 0.02
    synthetic_data = base_data + noise

    # Ensure values are in valid range [0, 1]
    synthetic_data = torch.clamp(synthetic_data, 0, 1)

    return synthetic_data

def create_segmentation_task(model, num_classes=10):
    """Create a segmentation task using terratorch"""
    task = SemanticSegmentationTask(
        model=model,
        model_args={
            'num_classes': 10,
        },
        ignore_index=-1,
        loss="ce",  # Cross-entropy loss
        class_weights=None
    )
    return task

def generate_land_cover_map(task, input_data):
    """Generate land cover classification using the segmentation task"""
    task.eval()

    with torch.no_grad():
        # Forward pass through the model
        outputs = task(input_data)

        # Get predictions
        if isinstance(outputs, dict):
            logits = outputs['out']
        else:
            logits = outputs

        # Convert to probabilities and class predictions
        #probabilities = torch.softmax(logits, dim=1, dtype=torch.float32)
        probabilities = torch.softmax(input_data, dim=1, dtype=torch.float32)
        land_cover = torch.argmax(probabilities, dim=1)

    return land_cover, probabilities

def create_change_detection_data(base_imagery, change_factor=0.3):
    """Create a second timestamp for change detection"""
    # Simulate changes in the imagery
    changed_imagery = base_imagery.clone()

    # Add some temporal changes
    height, width = base_imagery.shape[-2:]

    # Simulate deforestation (vegetation -> bare soil)
    change_mask = torch.rand(height, width) < change_factor
    for i in range(height):
        for j in range(width):
            if change_mask[i, j]:
                # Change vegetation pixels to bare soil signature
                if base_imagery[0, 2, i, j] > 0.3:  # High NIR (vegetation)
                    changed_imagery[0, :, i, j] = torch.tensor([0.12, 0.16, 0.22, 0.15, 0.35, 0.40])

    # Add temporal noise
    temporal_noise = torch.randn_like(changed_imagery) * 0.01
    changed_imagery = changed_imagery + temporal_noise
    changed_imagery = torch.clamp(changed_imagery, 0, 1)

    return changed_imagery

def save_as_geotiff(data, output_path, bounds=(-74.1, 40.7, -73.9, 40.8)):
    """Save data as GeoTIFF with spatial reference"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    # Handle different data shapes
    if len(data.shape) == 4:  # [batch, channels, height, width]
        data = data[0]  # Take first batch
    elif len(data.shape) == 3 and data.shape[0] > 10:  # Likely [channels, height, width]
        pass
    elif len(data.shape) == 3:  # [batch, height, width] - classification output
        data = data[0]  # Take first batch
        data = np.expand_dims(data, 0)  # Add channel dimension

    # Create geospatial transform
    transform = from_bounds(*bounds, data.shape[-1], data.shape[-2])

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0] if len(data.shape) > 2 else 1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        if len(data.shape) > 2:
            for i in range(data.shape[0]):
                dst.write(data[i], i+1)
        else:
            dst.write(data, 1)

def main():
    """Main execution function"""
    print("Loading Prithvi model via terratorch...")

    try:
        # Load model
        model = load_prithvi_model()

        # Create segmentation task
        task = create_segmentation_task(model, num_classes=6)

        print("Generating synthetic satellite imagery...")

        # Create synthetic input data
        input_data = create_synthetic_input_data(
            batch_size=1,
            channels=3,
            height=224,
            width=224
        )

        print("Generating land cover classification...")

        # Generate land cover map
        land_cover, probabilities = generate_land_cover_map(task, input_data)

        print("Creating temporal data for change detection...")

        # Create second timestamp
        t2_imagery = create_change_detection_data(input_data, change_factor=0.2)

        # Generate land cover for second timestamp
        land_cover_t2, probs_t2 = generate_land_cover_map(task, t2_imagery)

        # Calculate change map
        change_map = (land_cover != land_cover_t2).float()

        print("Saving outputs as GeoTIFF files...")

        # Save outputs
        bounds = (-74.1, 40.7, -73.9, 40.8)  # NYC area bounds

        save_as_geotiff(input_data, "synthetic_satellite_t1.tif", bounds)
        save_as_geotiff(t2_imagery, "synthetic_satellite_t2.tif", bounds)
        save_as_geotiff(land_cover, "land_cover_t1.tif", bounds)
        save_as_geotiff(land_cover_t2, "land_cover_t2.tif", bounds)
        save_as_geotiff(change_map, "change_detection_map.tif", bounds)

        print("Creating visualizations...")

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Time 1 RGB
        rgb_t1 = input_data[0, [2,1,0]].permute(1,2,0).cpu().numpy()
        rgb_t1 = np.clip(rgb_t1, 0, 1)
        axes[0,0].imshow(rgb_t1)
        axes[0,0].set_title("T1: RGB Composite")
        axes[0,0].axis('off')

        # Time 2 RGB
        rgb_t2 = t2_imagery[0, [2,1,0]].permute(1,2,0).cpu().numpy()
        rgb_t2 = np.clip(rgb_t2, 0, 1)
        axes[0,1].imshow(rgb_t2)
        axes[0,1].set_title("T2: RGB Composite")
        axes[0,1].axis('off')

        # Change map
        im_change = axes[0,2].imshow(change_map[0].cpu().numpy(), cmap='Reds')
        axes[0,2].set_title("Change Detection")
        axes[0,2].axis('off')
        plt.colorbar(im_change, ax=axes[0,2])

        # Land cover T1
        im_lc1 = axes[1,0].imshow(land_cover[0].cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
        axes[1,0].set_title("Land Cover T1")
        axes[1,0].axis('off')
        plt.colorbar(im_lc1, ax=axes[1,0])

        # Land cover T2
        im_lc2 = axes[1,1].imshow(land_cover_t2[0].cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
        axes[1,1].set_title("Land Cover T2")
        axes[1,1].axis('off')
        plt.colorbar(im_lc2, ax=axes[1,1])

        # NIR difference
        #nir_diff = t2_imagery[0,3] - input_data[0,3]  # NIR band difference
        #im_nir = axes[1,2].imshow(nir_diff.cpu().numpy(), cmap='RdBu_r')
        #axes[1,2].set_title("NIR Difference (T2-T1)")
        #axes[1,2].axis('off')
        #plt.colorbar(im_nir, ax=axes[1,2])

        plt.tight_layout()
        plt.savefig("terratorch_analysis_results.png", dpi=300, bbox_inches='tight')
        plt.show()

        print("Analysis complete! Generated files:")
        print("- synthetic_satellite_t1.tif")
        print("- synthetic_satellite_t2.tif")
        print("- land_cover_t1.tif")
        print("- land_cover_t2.tif")
        print("- change_detection_map.tif")
        print("- terratorch_analysis_results.png")

    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
