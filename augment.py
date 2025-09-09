import os
import torch
import SimpleITK as sitk
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandCoarseDropoutd, ScaleIntensityRanged,
    ToTensord, Compose
)
from monai.data import Dataset, DataLoader

# =============================
# STEP 1: Setup paths
# =============================
data_dir = "Preprocessed_Volumes/"       # input .nii volumes
output_root = "Augmented_Nifti_geom/"         # output folder
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
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    RandFlipd, RandRotate90d, RandAffined,
    ScaleIntensityRanged, RandScaleIntensityd, RandShiftIntensityd, 
    RandGaussianNoised, RandGaussianSmoothd, 
    RandCoarseDropoutd, RandAdjustContrastd, RandHistogramShiftd,
    ToTensord
)

train_transforms = Compose([
    # ---- Loading ----
    LoadImaged(keys=["image"], reader="ITKReader", image_only=False),
    EnsureChannelFirstd(keys=["image"]),

    # ---- Geometric ---- (equivalent to flips/rotations/zoom)
    RandFlipd(keys=["image"], prob=0.3, spatial_axis=[0, 1, 2]),   # like HorizontalFlip / Transpose
    RandRotate90d(keys=["image"], prob=0.3, max_k=3),              # random 90° rotations
    RandAffined(                                                   # minor zoom/translation/rotation
        keys=["image"], prob=0.2,
        rotate_range=(0.1, 0.1, 0.1),     # small random rotations
        scale_range=(0.02, 0.02, 0.02),   # ~2% zoom
        mode="bilinear"
    ),

    # ---- Intensity / Contrast ---- (equivalent to CLAHE, Hue/Sat, Gamma)
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1400, a_max=400,  # lung CT window
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.9, 1.1)),  # like RandomGamma
    RandHistogramShiftd(keys=["image"], num_control_points=3, prob=0.2),  # intensity warping
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),          # scale pixel values
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),          # shift brightness

    # ---- Robustness ---- (equivalent to Blur, Noise, Dropout)
    RandGaussianNoised(keys=["image"], prob=0.3),                        # like GaussianNoise
    RandGaussianSmoothd(keys=["image"], prob=0.2),                       # like GaussianBlur
    RandCoarseDropoutd(                                                  # like Cutout / ZoomBlur robustness
        keys=["image"], holes=5, spatial_size=(20, 20, 5), fill_value=0, prob=0.3
    ),

    # ---- Final ----
    ToTensord(keys=["image"]),
])


dataset = Dataset(data=data_dicts, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# =============================
# STEP 4: Apply augmentations and save
# =============================
for idx, batch in enumerate(dataloader):
    meta_dict = batch["image_meta_dict"]
    file_path = meta_dict["filename_or_obj"][0]
    series_name = batch["series_name"][0]

    series_output_dir = os.path.join(output_root, series_name)
    os.makedirs(series_output_dir, exist_ok=True)

    for aug_idx in range(1, num_augmentations + 1):
        image = batch["image"][0]  # tensor
        image_np = image.squeeze().cpu().numpy()  # shape: [z, y, x]

        if image_np.ndim == 4:  # remove channel dim if present
            image_np = image_np[0]

        # transpose axes to match SITK (X, Y, Z)
        image_np = image_np.transpose(2, 1, 0)

        sitk_img = sitk.GetImageFromArray(image_np)
        ref_img = sitk.ReadImage(file_path)
        sitk_img.SetSpacing(ref_img.GetSpacing())
        sitk_img.SetOrigin(ref_img.GetOrigin())
        sitk_img.SetDirection(ref_img.GetDirection())

        out_path = os.path.join(
            series_output_dir,
            os.path.basename(file_path).replace(".nii", f"_aug{aug_idx}.nii")
        )
        sitk.WriteImage(sitk_img, out_path)

        # =============================
        # Print augmentation log
        # =============================
        print(f"✅ Saved {out_path}")
        print(f"   Applied augmentations:")
        print(f"     - Random Flip (axes=[0,1,2]) prob=0.5")
        print(f"     - Random Rotate90 prob=0.5")
        print(f"     - HU windowing: [-1000, 400] → [0,1]")
        print(f"     - Random Intensity Scale prob=0.5")
        print(f"     - Random Intensity Shift prob=0.5")
        print(f"     - Gaussian Noise prob=0.3")
        print(f"     - Gaussian Smooth prob=0.2")
        print(f"     - Coarse Dropout prob=0.3 (5 holes, size=(20,20,5))")
