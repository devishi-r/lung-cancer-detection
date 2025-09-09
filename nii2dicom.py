import os
import SimpleITK as sitk

# Input and output paths
nii_root = "Augmented_Nifti_geom/"
dcm_root = "Augmented_DICOM_geom/"
os.makedirs(dcm_root, exist_ok=True)

# Recursively find .nii files
nii_files = []
for root, _, files in os.walk(nii_root):
    for f in files:
        if f.endswith(".nii"):
            nii_files.append(os.path.join(root, f))

print(f"Found {len(nii_files)} NIfTI files to convert.")

for nii_path in nii_files:
    # Load NIfTI
    img = sitk.ReadImage(nii_path)

    # Extract numpy array
    img_array = sitk.GetArrayFromImage(img)  # [slices, rows, cols]

    # Create output folder for this series
    rel_path = os.path.relpath(nii_path, nii_root)
    series_name = os.path.splitext(os.path.basename(nii_path))[0]
    series_dir = os.path.join(dcm_root, series_name)
    os.makedirs(series_dir, exist_ok=True)

    # Write one DICOM file per slice
    for i in range(img_array.shape[0]):
        # Convert float tensor to int16 for DICOM compatibility
        slice_np = img_array[i]
        slice_np = slice_np.astype('int16')
        slice_img = sitk.GetImageFromArray(slice_np)

        # ✅ Spacing and origin
        slice_img.SetSpacing(img.GetSpacing()[:2])
        slice_img.SetOrigin(img.GetOrigin()[:2])

        # ✅ Safe 2D direction
        dir3d = img.GetDirection()
        slice_dir = [dir3d[0], dir3d[1], dir3d[3], dir3d[4]]
        if slice_dir[0]*slice_dir[3] - slice_dir[1]*slice_dir[2] == 0:
            slice_dir = [1,0,0,1]  # fallback to identity if determinant=0
        slice_img.SetDirection(slice_dir)

        # Minimal DICOM metadata
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        dcm_path = os.path.join(series_dir, f"slice_{i:03d}.dcm")
        writer.SetFileName(dcm_path)
        writer.Execute(slice_img)

    print(f"✅ Converted {nii_path} → {series_dir}/")
