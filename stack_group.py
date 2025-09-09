import os
import re
import SimpleITK as sitk
from collections import defaultdict

# =============================
# CONFIG
# =============================
data_dir = "Data/"     # input root with DICOMs
output_dir = "Preprocessed_Volumes"  # where .nii will be saved
os.makedirs(output_dir, exist_ok=True)

# =============================
# HELPER: Sanitize filenames
# =============================
def sanitize_filename(name):
    name = re.sub(r"[^\w\-]", "_", name)   # replace spaces/special chars
    name = re.sub(r"_+", "_", name)        # collapse underscores
    return name.lstrip("_")                # avoid leading _

# =============================
# STEP 1: Group DICOMs into stacks
# =============================
def get_stacks(root_dir):
    """
    Recursively group DICOM slices into stacks for multiple patients.
    Assumes folder structure: patient -> study -> series -> dicom files.
    Returns: {(patient, series_path, stack_id): [slice_paths]}
    """
    stack_dict = {}
    stack_count_per_patient = defaultdict(int)

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

                # Collect DICOMs
                dicom_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
                if not dicom_files:
                    continue

                stacks = {}
                for f in dicom_files:
                    match = re.match(r"(\d+)-\d+", f)
                    if not match:
                        continue
                    stack_id = int(match.group(1))
                    stacks.setdefault(stack_id, []).append(os.path.join(series_path, f))

                for stack_id, slices in stacks.items():
                    slices.sort()
                    key = (patient_folder, series_path, stack_id)
                    stack_dict[key] = slices
                    stack_count_per_patient[patient_folder] += 1

    return stack_dict, stack_count_per_patient

# =============================
# STEP 2: Convert to NIfTI
# =============================
def convert_stack_to_volume(stack_files, output_path):
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(stack_files)
        image = reader.Execute()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(image, output_path)

        print(f"\n✅ Saved: {output_path} ({len(stack_files)} slices)")
        print("   Origin   :", image.GetOrigin())
        print("   Spacing  :", image.GetSpacing())
        print("   Direction:", image.GetDirection())
        print("   Size     :", image.GetSize())

    except Exception as e:
        print(f"❌ Failed {output_path}: {e}")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    stack_dict, stack_count_per_patient = get_stacks(data_dir)
    print(f"📦 Found {len(stack_dict)} stacks in total.\n")

    for patient, count in stack_count_per_patient.items():
        print(f"   📁 {patient}: {count} stacks")

    for (patient, series_path, stack_id), stack_files in stack_dict.items():
        series_name = sanitize_filename(os.path.basename(series_path.rstrip(os.sep)))
        out_name = f"{series_name}_stack{stack_id}.nii"
        out_path = os.path.join(output_dir, patient, out_name)
        convert_stack_to_volume(stack_files, out_path)
