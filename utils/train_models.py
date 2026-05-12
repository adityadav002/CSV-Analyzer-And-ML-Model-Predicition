import pandas as pd
import numpy as np

# Train Test Split
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

def train_models(df, target_column, result):
    if target_column is None:
        print("Target column is None")
        return

    if target_column not in df.columns:
        print("Target column not found in dataframe")
        return
    
    problem_type = result["problem_type"]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    label_encoder = LabelEncoder()

    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = label_encoder.fit_transform(X[column].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "XGBoost Regressor": XGBRegressor()
    }

    results_list = []

    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Store Results
        results_list.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })

    best_model = max(results_list, key=lambda x: x["R2 Score"])
    print("\nCompare Models\n")

    for model_result in results_list:
        print(f"Model: {model_result['Model']}")
        print(f"RMSE: {model_result['RMSE']}")
        print(f"R2 Score: {model_result['R2 Score']}")
        print("-" * 40)

    print("\nBest Model")
    print(f"Model Name: {best_model['Model']}")
    print(f"R2 Score: {best_model['R2 Score']}")

    return {
        "all_results": results_list,
        "best_model": best_model
    }