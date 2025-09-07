import pandas as pd
import os

# ----------------------------
# 1. Load metadata
# ----------------------------
def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the metadata.csv into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df


# ----------------------------
# 2. Filter out secondary storage
# ----------------------------
def filter_secondary(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the dataframe into primary (CT/PET) and secondary storage.
    Uses 'SOP Class Name' to identify scan type.
    """
    primary_df = df[df["SOP Class Name"].isin(
        ["CT Image Storage", "Positron Emission Tomography Image Storage"]
    )]
    secondary_df = df[~df["SOP Class Name"].isin(
        ["CT Image Storage", "Positron Emission Tomography Image Storage"]
    )]
    return primary_df, secondary_df


# ----------------------------
# 3. Count PET vs CT vs Secondary (by number of images)
# ----------------------------
def count_all_modalities(df: pd.DataFrame) -> pd.Series:
    """Count total number of DICOM files for CT, PET, and Secondary."""
    counts = {
        "CT": df.loc[df["SOP Class Name"] == "CT Image Storage", "Number of Images"].sum(),
        "PET": df.loc[df["SOP Class Name"] == "Positron Emission Tomography Image Storage", "Number of Images"].sum(),
        "Secondary": df.loc[~df["SOP Class Name"].isin(
            ["CT Image Storage", "Positron Emission Tomography Image Storage"]
        ), "Number of Images"].sum(),
    }
    return pd.Series(counts)


# ----------------------------
# 4. Count per category (CT vs PET by images)
# ----------------------------
def extract_category(subject_id: str) -> str:
    """Extract category letter (A, B, E, G) from Subject ID like 'Lung_Dx-A0002'."""
    try:
        return subject_id.split("-")[1][0]  # Take the first letter after 'Lung_Dx-'
    except Exception:
        return "Unknown"


def count_by_category_and_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Count CT vs PET per category (by number of images)."""
    df = df.copy()
    df["Category"] = df["Subject ID"].apply(extract_category)
    return df.groupby(["Category", "SOP Class Name"])["Number of Images"].sum().unstack(fill_value=0)


# ----------------------------
# 5. Count unique UIDs
# ----------------------------
def count_unique_uids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of unique UIDs per class (CT vs PET), both overall and per category.
    """
    df = df.copy()
    df["Category"] = df["Subject ID"].apply(extract_category)

    # Overall unique UID counts
    overall = df.groupby("SOP Class Name")["Series UID"].nunique()

    # Per-category unique UID counts
    per_category = df.groupby(["Category", "SOP Class Name"])["Series UID"].nunique().unstack(fill_value=0)

    return overall, per_category


# ----------------------------
# 6. Main execution
# ----------------------------
def main(csv_path: str):
    # Load data
    df = load_metadata(csv_path)

    # Show total counts before filtering
    print("=== Total DICOM Files (Before Filtering) ===")
    total_counts = count_all_modalities(df)
    print(total_counts.to_string(), "\n")

    # Per-category counts before filtering
    print("=== Per-Category CT vs PET Counts (Before Filtering) ===")
    before_cat = count_by_category_and_modality(df)
    print(before_cat.to_string(), "\n")

    # Unique UIDs before filtering
    print("=== Unique UIDs per Class (Before Filtering) ===")
    overall_uids, cat_uids = count_unique_uids(df)
    print("Overall:\n", overall_uids.to_string(), "\n")
    print("Per Category:\n", cat_uids.to_string(), "\n")

    # Filter secondary storage
    primary_df, secondary_df = filter_secondary(df)

    # Show counts after filtering
    print("=== Remaining DICOM Files (After Removing Secondary Storage) ===")
    remaining_counts = count_all_modalities(primary_df)
    print(remaining_counts.drop("Secondary").to_string(), "\n")

    # Per-category counts after filtering
    print("=== Per-Category CT vs PET Counts (After Filtering) ===")
    after_cat = count_by_category_and_modality(primary_df)
    print(after_cat.to_string(), "\n")

    # Unique UIDs after filtering
    print("=== Unique UIDs per Class (After Filtering) ===")
    overall_uids, cat_uids = count_unique_uids(primary_df)
    print("Overall:\n", overall_uids.to_string(), "\n")
    print("Per Category:\n", cat_uids.to_string(), "\n")

    # Deleted files
    deleted_count = secondary_df["Number of Images"].sum()
    print("=== Deleted Files (Secondary Storage) ===")
    print(f"Total deleted DICOM files: {deleted_count}")


if __name__ == "__main__":
    # Example usage:
    main("metadata.csv")
