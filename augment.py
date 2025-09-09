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
output_root = "Augmented_Nifti_geom/"    # output folder
num_augmentations = 1                    # number of augmented versions per stack
os.makedirs(output_root, exist_ok=True)

# =============================
# STEP 2: Build dataset entries
# =============================
nii_files = []
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.endswith(".nii"):
            nii_files.append(os.path.join(root, f))

data_dicts = []
for f in nii_files:
    filename = os.path.basename(f)

    # ✅ NEW: extract patient folder name (first-level under Preprocessed_Volumes)
    rel_path = os.path.relpath(f, data_dir)
    patient_name = rel_path.split(os.sep)[0]  

    # keep series name logic for unique filenames
    series_name = "_".join(filename.split("_")[:-1]) if "_stack" in filename else filename.replace(".nii", "")

    data_dicts.append({
        "image": f,
        "series_name": series_name,
        "patient_name": patient_name   # ✅ store patient for grouping
    })

# =============================
# STEP 3: Define augmentation pipeline
# =============================
train_transforms = Compose([
    LoadImaged(keys=["image"], reader="ITKReader", image_only=False),
    EnsureChannelFirstd(keys=["image"]),

    # ---- Geometric ----
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),

    # ---- Intensity / Windowing ----
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1400, a_max=400,  # HU lung window
        b_min=-1400, b_max=400,
        clip=True
    ),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3),

    # ---- Robustness ----
    RandGaussianSmoothd(keys=["image"], prob=0.2),
    RandCoarseDropoutd(keys=["image"], holes=5, spatial_size=(20, 20, 5), fill_value=0, prob=0.3),

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
    patient_name = batch["patient_name"][0]  # ✅ get patient folder

    # ✅ NEW: save under patient folder
    patient_output_dir = os.path.join(output_root, patient_name)
    os.makedirs(patient_output_dir, exist_ok=True)

    for aug_idx in range(1, num_augmentations + 1):
        image = batch["image"][0]
        image_np = image.squeeze().cpu().numpy()

        if image_np.ndim == 4:  # remove channel dim if present
            image_np = image_np[0]

        image_np = image_np.transpose(2, 1, 0)

        sitk_img = sitk.GetImageFromArray(image_np)
        ref_img = sitk.ReadImage(file_path)
        sitk_img.SetSpacing(ref_img.GetSpacing())
        sitk_img.SetOrigin(ref_img.GetOrigin())
        sitk_img.SetDirection(ref_img.GetDirection())

        # ✅ filename keeps series name but saved under patient folder
        out_path = os.path.join(
            patient_output_dir,
            os.path.basename(file_path).replace(".nii", f"_{series_name}_aug{aug_idx}.nii")
        )
        sitk.WriteImage(sitk_img, out_path)

        print(f"✅ Saved {out_path}")
        print(f"   Applied augmentations: Flip, Rotate90, HU window, Intensity Scale/Shift, Noise, Smooth, Dropout")
