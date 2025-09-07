import os
import SimpleITK as sitk
import numpy as np

# =============================
# Config
# =============================
data_dir = r"D:\Lung PET data\manifest-1608669183333\Lung-PET-CT-Dx\Lung_Dx-E0001"  # change if needed

# =============================
# Helper: Get affine matrix
# =============================
def get_affine(image):
    """
    Build a 4x4 affine matrix from a SimpleITK image.
    """
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    origin = np.array(image.GetOrigin())

    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine

# =============================
# Main loop
# =============================
print(f"Scanning folder: {data_dir}\n")

for root, _, files in os.walk(data_dir):
    dicom_files = [f for f in files if f.endswith(".dcm")]
    if not dicom_files:
        continue

    print(f"📂 Folder={os.path.basename(root)} → {len(dicom_files)} DICOMs")
    for idx, f in enumerate(sorted(dicom_files), 1):
        path = os.path.join(root, f)
        try:
            img = sitk.ReadImage(path)
            affine = get_affine(img)
            print(f"\n   • File {idx}: {f}")
            print(np.array2string(affine, precision=3, suppress_small=True))
        except Exception as e:
            print(f"   ⚠️ Skipping {f}: {e}")

    print("-" * 80)
