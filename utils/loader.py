import pandas as pd
import os

def load_dataset(filename):
    path = os.path.join("datasets", filename)
    df = pd.read_csv(path)

    if "fraud_detected" in df.columns:
        df["fraud_detected"] = (
            df["fraud_detected"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df[
            df["fraud_detected"] != "unknown"
        ]

    date_keywords = ["date", "time", "year"]
    for column in df.columns:
        cleaned_col = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(",", "", regex=False)
        )

        converted_numeric = pd.to_numeric(cleaned_col, errors="coerce")
        success_ratio = converted_numeric.notnull().sum() / len(df)

        if success_ratio > 0.8:
            df[column] = converted_numeric

        if any(keyword in column.lower() for keyword in date_keywords):
            try:
                converted_date = pd.to_datetime(df[column], errors="coerce")
                date_success_ratio = converted_date.notnull().sum() / len(df)
                if date_success_ratio > 0.8:
                    df[column] = converted_date
            except Exception:
                pass


    SKIP_KEYWORDS = ["url", "link", "image", "img", "website", "name", "gender", "age"]

    for column in df.columns:
        if not pd.api.types.is_string_dtype(df[column]):
            continue

        col_lower = column.lower()

        if any(kw in col_lower for kw in ["url", "link", "image", "img", "website"]):
            df[column] = df[column].astype(str).str.strip()

        elif not any(kw in col_lower for kw in ["name", "gender"]):
            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
            )

    boolean_mapping = {
        "true": "yes",  "yes": "yes", "y": "yes", "1": "yes", "t": "yes",
        "false": "no",  "no": "no",   "n": "no",  "0": "no",  "f": "no",
        "none": "unknown", "null": "unknown", "nan": "unknown",
        "unknown": "unknown", "na": "unknown", "n/a": "unknown",
    }

    for column in df.columns:
        if not pd.api.types.is_string_dtype(df[column]):
            continue

        unique_values = (
            df[column].dropna().astype(str).str.lower().unique()
        )

        if len(unique_values) == 0:
            continue

        boolean_like_count = sum(v in boolean_mapping for v in unique_values)

        if boolean_like_count / len(unique_values) >= 0.7:
            df[column] = (
                df[column].astype(str).str.lower().replace(boolean_mapping)
            )

    # =====================================================
    # Gender normalization
    # =====================================================

    gender_mapping = {
        "m": "male",   "ma": "male",  "mal": "male", "male": "male",
        "man": "male", "boy": "male",
        "f": "female", "fe": "female", "fem": "female", "fema": "female",
        "femal": "female", "female": "female", "woman": "female", "girl": "female",
        "nonbinary": "non-binary", "non binary": "non-binary", "nb": "non-binary",
        "other": "other",
        "none": "unknown", "unknown": "unknown", "nan": "unknown", "null": "unknown",
    }

    for column in df.columns:
        if "gender" in column.lower():
            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace(gender_mapping)
            )

    # =====================================================
    # Age cleaning
    # =====================================================

    for column in df.columns:
        if "age" in column.lower() and pd.api.types.is_numeric_dtype(df[column]):
            df = df[(df[column] >= 0) & (df[column] <= 125)]

    # =====================================================
    # Remove constant columns
    # =====================================================

    df.drop(
        columns=[col for col in df.columns if df[col].nunique() <= 1],
        inplace=True,
    )

    # =====================================================
    # Handle missing values
    # =====================================================

    row_threshold = df.shape[1] * 0.5
    df = df.dropna(thresh=row_threshold)

    cols_too_empty = [
        col for col in df.columns
        if df[col].isnull().sum() / len(df) > 0.5
    ]
    df.drop(columns=cols_too_empty, inplace=True)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_val = df[column].mode()
            if not mode_val.empty:
                df[column] = df[column].fillna(mode_val[0])

    # =====================================================
    # Remove duplicates
    # FIX: normalise only for comparison; don't overwrite data
    # =====================================================

    dedup_df = df.copy()
    for column in dedup_df.columns:
        if pd.api.types.is_string_dtype(dedup_df[column]):
            dedup_df[column] = (
                dedup_df[column]
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    duplicate_mask = dedup_df.duplicated()
    print("Duplicate Rows Found:", duplicate_mask.sum())
    df = df[~duplicate_mask]

    # =====================================================
    # Name formatting  (AFTER dedup so casing is preserved)
    # =====================================================

    for column in df.columns:
        if "name" in column.lower():
            df[column] = (
                df[column]
                .astype(str)
                .str.title()
                .str.replace(r"\s+", " ", regex=True)
            )

    # =====================================================
    # Save cleaned dataset
    # =====================================================

    output_path = os.path.join("outputs", f"cleaned_{filename}")
    df.to_csv(output_path, index=False)

    return df