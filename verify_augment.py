import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def verify_augmentation(original_path, augmented_path):
    # Load images
    orig = sitk.ReadImage(original_path)
    aug = sitk.ReadImage(augmented_path)
    
    # Check metadata
    print("Original spacing:", orig.GetSpacing(), "Augmented spacing:", aug.GetSpacing())
    print("Original size   :", orig.GetSize(), "Augmented size   :", aug.GetSize())
    print("Original origin :", orig.GetOrigin(), "Augmented origin :", aug.GetOrigin())
    
    # Convert to numpy
    orig_np = sitk.GetArrayFromImage(orig)
    aug_np = sitk.GetArrayFromImage(aug)
    
    # Check intensity stats
    print("Original intensity: min", orig_np.min(), "max", orig_np.max(), "mean", orig_np.mean())
    print("Augmented intensity: min", aug_np.min(), "max", aug_np.max(), "mean", aug_np.mean())
    
    # Visual comparison of mid-slice
    mid_slice = orig_np.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(orig_np[mid_slice], cmap="gray")
    axes[0].set_title("Original mid-slice")
    axes[1].imshow(aug_np[mid_slice], cmap="gray")
    axes[1].set_title("Augmented mid-slice")
    for ax in axes:
        ax.axis("off")
    plt.show()

# Example usage
original_file = "Preprocessed_Volumes/2_000000-A_phase_5mm_Stnd_SS50-58188_stack1.nii"
augmented_file = "Augmented_Nifti/2_000000-A_phase_5mm_Stnd_SS50-58188/2_000000-A_phase_5mm_Stnd_SS50-58188_stack1_aug1.nii"

verify_augmentation(original_file, augmented_file)
