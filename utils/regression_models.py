import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


def train_regression_models(df, target_column, result):

    X = df.drop(columns=[target_column]).copy()

    y = df[target_column].copy()

    # =========================
    # ENCODE CATEGORICAL COLUMNS
    # =========================
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        le = LabelEncoder()

        X[column] = le.fit_transform(
            X[column].astype(str)
        )

    # =========================
    # SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # MODELS
    # =========================
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "XGBoost Regressor": XGBRegressor()
    }

    all_results = []

    # =========================
    # TRAIN MODELS
    # =========================
    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = round(
            np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
            2
        )

        r2 = round(
            r2_score(y_test, predictions),
            4
        )

        model_result = {
            "Model": name,
            "RMSE": rmse,
            "R2 Score": r2
        }

        all_results.append(model_result)

    # =========================
    # BEST MODEL
    # =========================
    best_model = max(
        all_results,
        key=lambda x: x["R2 Score"]
    )

    # =========================
    # PRINT OUTPUT
    # =========================
    print("\nCompare Models\n")

    for result in all_results:

        print(f"Model: {result['Model']}")
        print(f"RMSE: {result['RMSE']}")
        print(f"R2 Score: {result['R2 Score']}")
        print("-" * 40)

    print("\nBest Model")
    print(best_model)

    return {
        "all_results": all_results,
        "best_model": best_model
    }