import os
import numpy as np
import SimpleITK as sitk
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.stats import wasserstein_distance
from augment import *

def load_dicom_series(dicom_files):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)  # [slices, H, W]

def compare_volumes(original_files, augmented_files):
    # Load series
    vol_orig = load_dicom_series(original_files).astype(np.float32)
    vol_aug = load_dicom_series(augmented_files).astype(np.float32)

    # Resize if shapes differ
    if vol_orig.shape != vol_aug.shape:
        min_slices = min(vol_orig.shape[0], vol_aug.shape[0])
        vol_orig = vol_orig[:min_slices]
        vol_aug = vol_aug[:min_slices]

    # --- Metrics ---
    mse = mean_squared_error(vol_orig, vol_aug)
    psnr = peak_signal_noise_ratio(vol_orig, vol_aug, data_range=vol_orig.max() - vol_orig.min())
    ssim = structural_similarity(vol_orig, vol_aug, data_range=vol_orig.max() - vol_orig.min(), channel_axis=None)

    # Histogram similarity (per slice, averaged)
    hist_sim = []
    for i in range(vol_orig.shape[0]):
        hist_orig, _ = np.histogram(vol_orig[i], bins=100, range=(vol_orig.min(), vol_orig.max()), density=True)
        hist_aug, _ = np.histogram(vol_aug[i], bins=100, range=(vol_orig.min(), vol_orig.max()), density=True)
        hist_sim.append(-wasserstein_distance(hist_orig, hist_aug))  # closer to 0 is better
    hist_sim = np.mean(hist_sim)

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "HistogramSim": hist_sim
    }

# ======================
# Example usage
# ======================
# Original and augmented file lists (from one UID)
original_files = sorted(class_e_dicom_series[0]["image"])  # first original case
augmented_dir = os.path.join(output_dir, "10-25-2007-NA-lung-83596", "2.000000-A phase 5mm Stnd SS50-58188", "2.000000-A phase 5mm Stnd SS50-58188_aug0")  # adjust as per your script
augmented_files = sorted([os.path.join(augmented_dir, f) for f in os.listdir(augmented_dir)])

results = compare_volumes(original_files, augmented_files)
print("🔍 Similarity Metrics:", results)
