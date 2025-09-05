import os
import glob
import torch
# import numpy as np
import SimpleITK as sitk
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised,
    ToTensord, Compose
)
from monai.data import Dataset, DataLoader
from getUID import *

# =============================
# STEP 2: Mount Google Drive
# =============================
# from google.colab import drive
# drive.mount('/content/drive')

# =============================
# STEP 3: Define Paths and Gather Class E Data
# =============================
# path = 'Lung-PET-CT-Dx/Lung_Dx-E0001/'
data_dir = "Lung-PET-CT-Dx/Lung_Dx-E0001/"
output_dir = "LungData_Augmented_Monai/E/"
os.makedirs(output_dir, exist_ok=True)


uid_dict = getUID_path(data_dir)  # {UID: (dicom_path, dicom_file)}

# Group dicoms by UID
series_dict = {}
for uid, (dicom_path, dicom_file) in uid_dict.items():
    series_dict.setdefault(uid, []).append(dicom_path)

# Build MONAI dataset entries
class_e_dicom_series = [{"image": sorted(paths)} for uid, paths in series_dict.items()]
print(f"✅ Found {len(class_e_dicom_series)} unique series (by UID).")



# Find all Class E cases
# class_e_folders = glob.glob(os.path.join(data_dir, "Lung_Dx-E*"))
# class_e_dicom_series = []

# for folder in class_e_folders:
#     dicom_files = glob.glob(os.path.join(folder, "**/*.dcm"), recursive=True)
#     if dicom_files:
#         class_e_dicom_series.append({"image": dicom_files})

print(f"✅ Found {len(class_e_dicom_series)} Class E cases.")

# =============================
# STEP 4: Define MONAI Augmentation Pipeline
# =============================
train_transforms = Compose([
    LoadImaged(keys=["image"], reader="PydicomReader", ensure_channel_first=True    ),
    EnsureChannelFirstd(keys=["image"]),  # Add channel for CNN compatibility
    # Orientationd(keys=["image"], axcodes="RAS"),  # Normalize orientation
    # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),  # Resample
    # RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    # RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3),
    ToTensord(keys=["image"])
])

dataset = Dataset(data=class_e_dicom_series, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# =============================
# STEP 5: Function to Save Augmented Volume as DICOM
# =============================
def save_as_dicom(volume_tensor, reference_dicom_files, output_folder, prefix="aug"):
    os.makedirs(output_folder, exist_ok=True)
    volume_np = volume_tensor.squeeze().cpu().numpy()
    
    if volume_np.ndim == 2:
        volume_np = volume_np[None, :, :]

    if volume_np.dtype != "int16":
        volume_np = volume_np.astype("int16")

    # Use reference metadata
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reference_dicom_files)
    reference_image = reader.Execute()

    # Create new SimpleITK image
    new_image = sitk.GetImageFromArray(volume_np)
    new_image.CopyInformation(reference_image)

    # Save as DICOM series
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for i in range(volume_np.shape[0]):
        dicom_path = os.path.join(output_folder, f"{prefix}_{i:03d}.dcm")
        writer.SetFileName(dicom_path)
        writer.Execute(new_image[:, :, i])

# =============================
# STEP 6: Generate Augmented Data
# =============================
num_augmentations = 5  # how many augmented versions per original case?

for idx, batch in enumerate(dataloader):
    print(f"Batch {idx} shape:", batch["image"].shape)
    original_files = class_e_dicom_series[idx]["image"]
    for aug_idx in range(num_augmentations):
        aug_image = batch["image"]  # already transformed
        aug_folder = os.path.join(output_dir, f"E_case{idx+1}_aug{aug_idx+1}")
        save_as_dicom(aug_image, original_files, aug_folder)
        print(f"✅ Saved augmented case {idx+1} - version {aug_idx+1}")