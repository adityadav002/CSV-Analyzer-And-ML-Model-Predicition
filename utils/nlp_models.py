import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score
)


def train_nlp_models(df, target_column, result):

    # =========================
    # FIND TEXT COLUMN
    # =========================
    text_column = None

    for column in df.columns:

        if column == target_column:
            continue

        if df[column].dtype == "object":

            avg_length = (
                df[column]
                .astype(str)
                .str.len()
                .mean()
            )

            if avg_length > 20:
                text_column = column
                break

    # =========================
    # FEATURES & TARGET
    # =========================
    X = df[text_column].astype(str)

    y = df[target_column].astype(str)

    # =========================
    # ENCODE TARGET
    # =========================
    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(y)

    # =========================
    # TF-IDF
    # =========================
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(X)

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
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
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