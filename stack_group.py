import os
import re
import SimpleITK as sitk

# =============================
# CONFIG
# =============================
data_dir = "Lung-PET-CT-Dx/Lung_Dx-E0001/"     # input root with DICOMs
output_dir = "Preprocessed_Volumes"            # where .nii will be saved
os.makedirs(output_dir, exist_ok=True)

# =============================
# HELPER: Sanitize filenames for Windows/NIfTI
# =============================
def sanitize_filename(name):
    # replace spaces and any non-alphanumeric characters with underscore
    name = re.sub(r"[^\w\-]", "_", name)
    # remove multiple underscores
    name = re.sub(r"_+", "_", name)
    # avoid leading dot
    name = name.lstrip("_")
    return name

# =============================
# STEP 1: Group DICOMs into stacks
# =============================
def get_stacks(data_dir):
    """
    Group DICOM slices into stacks based on the filename prefix before '-'.
    Returns { (series_path, stack_id): [slice_paths] }
    """
    stack_dict = {}
    for root, _, files in os.walk(data_dir):
        dicom_files = [f for f in files if f.endswith(".dcm")]
        if not dicom_files:
            continue

        stacks = {}
        for f in dicom_files:
            match = re.match(r"(\d+)-\d+", f)
            if not match:
                continue
            stack_id = int(match.group(1))
            stacks.setdefault(stack_id, []).append(os.path.join(root, f))

        for stack_id, slices in stacks.items():
            slices.sort()
            stack_dict[(root, stack_id)] = slices

    return stack_dict

# =============================
# STEP 2: Convert stack to 3D volume and save as .nii
# =============================
def convert_stack_to_volume(stack_files, output_path):
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(stack_files)
        image = reader.Execute()

        # ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # write as .nii (no compression)
        sitk.WriteImage(image, output_path)

        print(f"\n✅ Saved volume: {output_path} ({len(stack_files)} slices)")
        print("   Origin   :", image.GetOrigin())
        print("   Spacing  :", image.GetSpacing())
        print("   Direction:", image.GetDirection())
        print("   Size     :", image.GetSize())

    except Exception as e:
        print(f"❌ Failed to process {output_path}: {e}")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    stack_dict = get_stacks(data_dir)
    print(f"📦 Found {len(stack_dict)} stacks in {data_dir}")

    for (series_path, stack_id), stack_files in stack_dict.items():
        raw_name = os.path.basename(series_path.rstrip(os.sep))
        series_name = sanitize_filename(raw_name)
        out_name = f"{series_name}_stack{stack_id}.nii"  # safe .nii
        out_path = os.path.join(output_dir, out_name)

        convert_stack_to_volume(stack_files, out_path)
