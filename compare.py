import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# =============================
# Setup
# =============================
orig_file = "Preprocessed_Volumes/2_000000-A_phase_5mm_Stnd_SS50-58188_stack1"   # <-- put one of your originals
aug_file  = "Augmented_Nifti_geom/2_000000-A_phase_5mm_Stnd_SS50-58188/2_000000-A_phase_5mm_Stnd_SS50-58188_stack1_aug1"  # <-- corresponding augmented

# =============================
# Load with SimpleITK
# =============================
orig_img = sitk.ReadImage(orig_file)
aug_img  = sitk.ReadImage(aug_file)

orig_np = sitk.GetArrayFromImage(orig_img)  # (Z, Y, X)
aug_np  = sitk.GetArrayFromImage(aug_img)

# =============================
# Metadata comparison
# =============================
print("---- Metadata Check ----")
print("Original spacing :", orig_img.GetSpacing(), " | Aug spacing :", aug_img.GetSpacing())
print("Original size    :", orig_img.GetSize(), "   | Aug size    :", aug_img.GetSize())
print("Original origin  :", orig_img.GetOrigin(), "  | Aug origin  :", aug_img.GetOrigin())
print("Original dir     :", orig_img.GetDirection(), " | Aug dir     :", aug_img.GetDirection())

print("\n---- Intensity Check ----")
print(f"Original intensity: min {orig_np.min():.2f}, max {orig_np.max():.2f}, mean {orig_np.mean():.2f}")
print(f"Augmented intensity: min {aug_np.min():.2f}, max {aug_np.max():.2f}, mean {aug_np.mean():.2f}")

# =============================
# Histograms
# =============================
plt.figure(figsize=(10,5))
plt.hist(orig_np.flatten(), bins=100, alpha=0.5, label="Original")
plt.hist(aug_np.flatten(), bins=100, alpha=0.5, label="Augmented")
plt.legend()
plt.title("Intensity Histograms")
plt.show()

# =============================
# Slice visualization
# =============================
z_mid = orig_np.shape[0] // 2  # pick mid slice
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(orig_np[z_mid,:,:], cmap="gray")
plt.title("Original mid-slice")

plt.subplot(1,2,2)
plt.imshow(aug_np[z_mid,:,:], cmap="gray")
plt.title("Augmented mid-slice")

plt.show()
