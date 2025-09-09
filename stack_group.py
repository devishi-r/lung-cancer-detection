import os
import re
import SimpleITK as sitk

# =============================
# CONFIG
# =============================
data_dir = "Data/"     # input root with DICOMs
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
def get_stacks(root_dir):
    """
    Recursively group DICOM slices into stacks for multiple patients.
    Assumes folder structure: patient -> study -> series -> dicom files.
    Returns: {(series_path, stack_id): [slice_paths]}
    """
    stack_dict = {}

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for study_folder in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_folder)
            if not os.path.isdir(study_path):
                continue

            for series_folder in os.listdir(study_path):
                series_path = os.path.join(study_path, series_folder)
                if not os.path.isdir(series_path):
                    continue

                # Collect all DICOM files in this series
                dicom_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
                if not dicom_files:
                    continue

                # Group slices by stack ID (from filename)
                stacks = {}
                for f in dicom_files:
                    match = re.match(r"(\d+)-\d+", f)
                    if not match:
                        continue
                    stack_id = int(match.group(1))
                    stacks.setdefault(stack_id, []).append(os.path.join(series_path, f))

                # Sort slices within each stack and add to main dict
                for stack_id, slices in stacks.items():
                    slices.sort()
                    stack_dict[(series_path, stack_id)] = slices

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
