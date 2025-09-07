import os
import torch
import SimpleITK as sitk
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    ToTensord, Compose
)
from monai.data import Dataset, DataLoader

# =============================
# STEP 1: Setup paths
# =============================
data_dir = "Preprocessed_Volumes/"       # input .nii volumes
output_root = "Augmented_Nifti/"         # output folder
num_augmentations = 1                    # number of augmented versions per stack
os.makedirs(output_root, exist_ok=True)

# =============================
# STEP 2: Build dataset entries
# =============================
nii_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nii")]

data_dicts = []
for f in nii_files:
    filename = os.path.basename(f)
    # everything before "_stackN"
    series_name = "_".join(filename.split("_")[:-1])
    data_dicts.append({"image": f, "series_name": series_name})

# =============================
# STEP 3: Define augmentation pipeline
# =============================
train_transforms = Compose([
    LoadImaged(keys=["image"], reader="ITKReader", image_only=False),  # load metadata too
    EnsureChannelFirstd(keys=["image"]),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3),
    ToTensord(keys=["image"]),
])

dataset = Dataset(data=data_dicts, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# =============================
# STEP 4: Apply augmentations and save
# =============================
for idx, batch in enumerate(dataloader):
    # MONAI keeps metadata in "image_meta_dict"
    meta_dict = batch["image_meta_dict"]
    file_path = meta_dict["filename_or_obj"][0]
    series_name = batch["series_name"][0]

    # create output folder mirroring original series
    series_output_dir = os.path.join(output_root, series_name)
    os.makedirs(series_output_dir, exist_ok=True)

    for aug_idx in range(1, num_augmentations + 1):
        image = batch["image"][0]  # tensor
        image_np = image.squeeze().cpu().numpy()  # shape: [z, y, x]

        # remove channel dim if present
        if image_np.ndim == 4:
            image_np = image_np[0]

        # transpose axes to match SITK (X, Y, Z)
        image_np = image_np.transpose(2, 1, 0)  # now shape is [X, Y, Z]

        # create SimpleITK image and copy original metadata
        sitk_img = sitk.GetImageFromArray(image_np)
        ref_img = sitk.ReadImage(file_path)
        sitk_img.SetSpacing(ref_img.GetSpacing())
        sitk_img.SetOrigin(ref_img.GetOrigin())
        sitk_img.SetDirection(ref_img.GetDirection())

        out_path = os.path.join(series_output_dir,
                                os.path.basename(file_path).replace(".nii", f"_aug{aug_idx}.nii"))
        sitk.WriteImage(sitk_img, out_path)
        print(f"✅ Saved {out_path}")
