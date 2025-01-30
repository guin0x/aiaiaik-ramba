import os
import torch
import numpy as np
from osgeo import gdal
from torch.utils.data import TensorDataset
from preprocessing.satellite_analysis_pre import *
from pathlib import Path

def prepare_dataset(train_val_test, year_target=5, nonwater_threshold=480000, 
                    nodata_value=-1, nonwater_value=0, dir_folders=Path('data/satellite/dataset'), 
                    collection='JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64):
    """
    Prepare the full dataset for training, validation, or testing in one step.
    
    Args:
        train_val_test (str): Specifies the dataset type: 'training', 'validation', or 'testing'.
        year_target (int): Year predicted after a sequence of input years (default is 5).
        nonwater_threshold (int): Minimum 'non-water' pixels required in input and target (default is 480,000).
        nodata_value (int): Pixel value for 'no-data' class (-1 for scaled, 0 otherwise).
        nonwater_value (int): Pixel value for 'non-water' class (0 for scaled, 1 otherwise).
        dir_folders (Path): Directory containing the satellite dataset folders.
        collection (str): Satellite image collection to process.
        scaled_classes (bool): Whether to scale pixel values to [-1, 1] (default is True).
        device (str): Device for computation ('cuda:0' or 'cpu').
        dtype (torch.dtype): Data type for tensor creation (default is torch.int64).

    Returns:
        TensorDataset: Ready-to-use dataset with inputs and targets.
    """
    def load_image_array(path):
        img = gdal.Open(str(path))
        img_array = img.ReadAsArray().astype(np.float32)
        if scaled_classes:
            img_array = img_array.astype(int)
            img_array[img_array == 0] = -1
            img_array[img_array == 1] = 0
            img_array[img_array == 2] = 1
        return img_array

    def get_image_paths(reach):
        folder = dir_folders / f"{collection}_{train_val_test}_r{reach}"
        return [path for path in folder.iterdir() if path.suffix == '.tif']

    def process_reach(reach):
        paths = get_image_paths(reach)
        all_years = list(range(1988, 1988 + len(paths)))  # All years based on available files
        images = [load_image_array(path) for path in paths]
        averages = [load_avg(train_val_test, reach, year, dir_averages=Path('data/satellite/averages')) 
                    for year in all_years]
        images = [np.where(img == nodata_value, avg, img) for img, avg in zip(images, averages)]
        used_years = []
        inputs, targets = [], []
        for i in range(len(images) - year_target):
            input_seq = images[i:i+year_target-1]
            target_img = images[i+year_target-1]
            if all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_seq) and \
            count_pixels(target_img, nonwater_value) < nonwater_threshold:
                inputs.append(input_seq)
                targets.append(target_img)
                input_years = list(range(1988 + i, 1988 + i + year_target - 1))
                target_year = 1988 + i + year_target - 1
                used_years.extend(input_years + [target_year])
        return inputs, targets

    # Combine data from all reaches
    combined_inputs, combined_targets = [], []
    for folder in dir_folders.iterdir():
        if train_val_test in folder.name:
            reach = int(folder.name.split('_r')[-1])
            inputs, targets = process_reach(reach)
            combined_inputs.extend(inputs)
            combined_targets.extend(targets)

    # Convert to tensors and create dataset
    input_tensor = torch.tensor(combined_inputs, dtype=dtype, device=device)
    target_tensor = torch.tensor(combined_targets, dtype=dtype, device=device)
    return TensorDataset(input_tensor, target_tensor)

def prepare_3d_dataset(train_val_test, year_target=5, nonwater_threshold=480000, 
                       nodata_value=-1, nonwater_value=0, dir_folders='data/satellite/dataset', 
                       collection='JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.float32):
    """
    Prepare the full dataset for training, validation, or testing with temporal sequences for 3D convolution.
    """
    def load_image_array(path):
        img = gdal.Open(path)
        img_array = img.ReadAsArray().astype(np.float32)
        if scaled_classes:
            img_array = img_array.astype(int)
            img_array[img_array == 0] = -1
            img_array[img_array == 1] = 0
            img_array[img_array == 2] = 1
        return img_array

    def get_image_paths(reach):
        folder = os.path.join(dir_folders, f"{collection}_{train_val_test}_r{reach}")
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]

    def process_reach(reach):
        paths = get_image_paths(reach)
        images = [load_image_array(path) for path in paths]
        averages = [load_avg(train_val_test, reach, year, dir_averages='data/satellite/averages') 
                    for year in range(1988, 1988 + len(images))]
        images = [np.where(img == nodata_value, avg, img) for img, avg in zip(images, averages)]
        inputs, targets = [], []
        for i in range(len(images) - year_target):
            input_seq = images[i:i+year_target-1]
            target_img = images[i+year_target-1]
            if all(count_pixels(img, nonwater_value) < nonwater_threshold for img in input_seq) and \
               count_pixels(target_img, nonwater_value) < nonwater_threshold:
                inputs.append(input_seq)
                targets.append(target_img)
        return inputs, targets

    # Combine data from all reaches
    combined_inputs, combined_targets = [], []
    for folder in os.listdir(dir_folders):
        if train_val_test in folder:
            reach = int(folder.split('_r')[-1])
            inputs, targets = process_reach(reach)
            combined_inputs.extend(inputs)
            combined_targets.extend(targets)

    # Prepare tensors for 3D convolution
    combined_inputs = np.array(combined_inputs)  # Shape: (N, T, H, W)
    combined_targets = np.array(combined_targets)  # Shape: (N, H, W)

    input_tensor = torch.tensor(combined_inputs[:, None, :, :, :], dtype=dtype, device=device)  # Shape: (N, 1, T, H, W)
    target_tensor = torch.tensor(combined_targets, dtype=dtype, device=device)  # Shape: (N, H, W)

    return TensorDataset(input_tensor, target_tensor)

def prepare_3d_dataset_with_ndvi(train_val_test, year_target=5, nonwater_threshold=480000, 
                                 nodata_value=-1, nonwater_value=0, binary_dir='data/satellite/dataset', 
                                 ndvi_dir='data/satellite/ndvi', collection='JRC_GSW1_4_MonthlyHistory', 
                                 scaled_classes=True, device='cuda:0', dtype=torch.float32):
    """
    Prepare the full dataset for training, validation, or testing with temporal sequences for 3D convolution, 
    including NDVI as an additional input channel.
    """
    def load_image_array_ndvi(path):
        img = gdal.Open(path)
        img_array = img.ReadAsArray().astype(np.float32)

        return img_array

    def load_image_array_binary(path):
        img = gdal.Open(path)
        img_array = img.ReadAsArray().astype(np.float32)

        if scaled_classes:
            img_array = img_array.astype(int)
            img_array[img_array == 0] = -1
            img_array[img_array == 1] = 0
            img_array[img_array == 2] = 1
        return img_array

    def get_binary_paths(reach):
        folder = os.path.join(binary_dir, f"{collection}_{train_val_test}_r{reach}")
        return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')])

    def get_ndvi_paths(reach):
        folder = os.path.join(ndvi_dir, f"{train_val_test}_r{reach}")
        return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')])

    def process_reach(reach):
        import matplotlib.pyplot as plt
                
        # Get paths for binary and NDVI datasets
        binary_paths = get_binary_paths(reach)
        ndvi_paths = get_ndvi_paths(reach)

        # binary_paths = binary_paths[:-1]

        # Ensure alignment of binary and NDVI datasets
        assert len(binary_paths) == len(ndvi_paths), "Binary and NDVI dataset lengths do not match!"

        # ndvi_images_before_rotation = np.array([load_image_array_ndvi(path) for path in ndvi_paths])
       
        # for i in range(len(binary_paths)):
        #     _ = preprocess_images(str(ndvi_paths[i]), reshape_img=True)

        binary_images = [load_image_array_binary(path) for path in binary_paths]
        ndvi_images = [load_image_array_ndvi(path) for path in ndvi_paths]
        
        # Load averages and replace nodata
        years = range(1988, 1988 + len(binary_images))
        binary_averages = [load_avg(train_val_test, reach, year, dir_averages='data/satellite/averages') for year in years]
        binary_images = [np.where(img == nodata_value, avg, img) for img, avg in zip(binary_images, binary_averages)]
        
        inputs, targets = [], []

        for i in range(len(binary_images) - year_target):
            # Prepare input sequences
            binary_seq = binary_images[i:i+year_target-1]
            ndvi_seq = ndvi_images[i:i+year_target-1]
   
            # # for loop here is to check/fix shapes
            # for j in range(len(ndvi_seq)):
                    
            #     if ndvi_seq[j].shape == (1001, 500):
            #         ndvi_seq[j] = ndvi_seq[j][:-1, :]
            #     if ndvi_seq[j].shape == (1001, 501):
            #         ndvi_seq[j] = ndvi_seq[j][:-1, :-1]
            #     if ndvi_seq[j].shape == (1000, 501):
            #         ndvi_seq[j] = ndvi_seq[j][:, :-1]

            input_seq = []

            for binary, ndvi in zip(binary_seq, ndvi_seq):
                try:
                    stacked_array = np.stack([binary, ndvi], axis=0)
                    input_seq.append(stacked_array)
                except Exception as e:
                    # Handle the exception, e.g., log it or skip the pair
                    print(f"Error stacking binary and NDVI: {e}")
                    # print(binary.shape, ndvi.shape)
                    plt.imshow(binary)
                    plt.show()
                    plt.imshow(ndvi)
                    plt.show()


            # Prepare target
            target_img = binary_images[i+year_target-1]

            # Apply pixel thresholds
            if all(count_pixels(img, nonwater_value) < nonwater_threshold for img in binary_seq) and \
               count_pixels(target_img, nonwater_value) < nonwater_threshold:
                inputs.append(np.stack(input_seq, axis=1))  # Combine temporal inputs
                targets.append(target_img)

        return inputs, targets
    # Combine data from all reaches
    total_folders = sum(1 for folder in os.listdir(binary_dir) if train_val_test in folder)

    combined_inputs_tensors = []
    combined_targets_tensors = []

    for k, folder in enumerate(os.listdir(binary_dir)):
        if train_val_test in folder:
            reach = int(folder.split('_r')[-1])
            inputs, targets = process_reach(reach)
            
            # Convert to tensors and append directly
            combined_inputs_tensors.extend(
                torch.tensor(inputs, dtype=torch.float32, device=device)
            )
            combined_targets_tensors.extend(
                torch.tensor(targets, dtype=torch.float32, device=device)
            )

    # Combine all tensors into a single tensor
    print("Finished processing folders.")
    input_tensor = torch.stack(combined_inputs_tensors)
    print("Finisehd stacking input tensors.")
    target_tensor = torch.stack(combined_targets_tensors)
    print("Finished stacking target tensors.")


    return TensorDataset(input_tensor, target_tensor)

