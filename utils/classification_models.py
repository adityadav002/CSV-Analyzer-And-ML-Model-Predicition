import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score
)


def train_classification_models(df, target_column, result):

    X = df.drop(columns=[target_column]).copy()

    y = df[target_column].copy()

    # =========================
    # ENCODE FEATURES
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
    # ENCODE TARGET
    # =========================
    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        y.astype(str)
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
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "SVM": SVC(),
        "XGBoost Classifier": XGBClassifier()
    }

    all_results = []

    # =========================
    # TRAIN MODELS
    # =========================
    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = round(
            accuracy_score(y_test, predictions) * 100,
            2
        )

        precision = round(
            precision_score(
                y_test,
                predictions,
                average="weighted"
            ) * 100,
            2
        )

        model_result = {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision
        }

        all_results.append(model_result)

    # =========================
    # BEST MODEL
    # =========================
    best_model = max(
        all_results,
        key=lambda x: x["Accuracy"]
    )

    # =========================
    # PRINT OUTPUT
    # =========================
    print("\nCompare Models\n")

    for result in all_results:

        print(f"Model: {result['Model']}")
        print(f"Accuracy: {result['Accuracy']}%")
        print(f"Precision: {result['Precision']}%")
        print("-" * 40)

    print("\nBest Model")
    print(best_model)

    return {
        "all_results": all_results,
        "best_model": best_model
    }