import pandas as pd

def detect_problem_type(df, target_column=None):    
    if target_column is None:
        return {
            "problem_type": "Clustering",
            "sub_type": None,
            "algorithms": ["K-Means", "DBSCAN", "Hierarchical Clustering"],
        }

    # =========================
    # SAFETY CHECK
    # =========================
    if target_column not in df.columns:
        return {"problem_type": "Unknown", "sub_type": None, "algorithms": []}
    
    target = df[target_column].dropna()

    if len(target) == 0:
        return {
            "problem_type": "Unknown",
            "sub_type": None,
            "algorithms": []
        }

    unique_values = target.nunique()

    # =========================
    # NLP DETECTION
    # =========================
    text_columns = []

    for column in df.columns:
        if column == target_column:
            continue
        if pd.api.types.is_object_dtype(df[column]):
            avg_length = (
                df[column]
                .astype(str)
                .str.len()
                .mean()
            )
            unique_ratio = (
                df[column].nunique() / len(df)
            )
            # Long text + many unique values
            if avg_length > 20 and unique_ratio > 0.5:
                text_columns.append(column)

    
            target_is_categorical = (
                pd.api.types.is_object_dtype(target)
                or pd.api.types.is_string_dtype(target)
                or pd.api.types.is_bool_dtype(target)
            )

            if len(text_columns) > 0 and target_is_categorical:
                return {
                    "problem_type": "NLP Classification",
                    "sub_type": "Text Classification",
                    "algorithms": [
                        "Naive Bayes",
                        "Logistic Regression",
                        "SVM"
                    ]
                }

    # =========================
    # OBJECT / CATEGORY / BOOL
    # =========================
    is_categorical = (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or pd.api.types.is_bool_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    )

    if is_categorical:
        sub_type = "Binary Classification" if unique_values == 2 else "Multiclass Classification"
        return {
            "problem_type": "Classification",
            "sub_type": sub_type,
            "algorithms": ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree", "SVM"],
        }

    # =========================
    # NUMERICAL TARGET
    # =========================
    if pd.api.types.is_numeric_dtype(target):

        if unique_values <= 20:
            sub_type = "Binary Classification" if unique_values == 2 else "Multiclass Classification"
            return {
                "problem_type": "Classification",
                "sub_type": sub_type,
                "algorithms": ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Decision Tree"],
            }

        return {
            "problem_type": "Regression",
            "sub_type": None,
            "algorithms": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "Decision Tree Regressor"],
        }

    return {"problem_type": "Unknown", "sub_type": None, "algorithms": []}