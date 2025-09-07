import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

nii_path = "Preprocessed_Volumes/2.000000-A phase 5mm Stnd SS50-58188_stack1.nii"

# Load
image = sitk.ReadImage(nii_path)

# Metadata
print("Origin   :", image.GetOrigin())
print("Spacing  :", image.GetSpacing())
print("Direction:", image.GetDirection())
print("Size     :", image.GetSize())

# Convert to numpy
array = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]
print("Array shape:", array.shape)

# Show mid-slice
mid = array.shape[0] // 2
plt.imshow(array[mid, :, :], cmap="gray")
plt.title(f"Mid-slice of {os.path.basename(nii_path)}")
plt.axis("off")
plt.show()
