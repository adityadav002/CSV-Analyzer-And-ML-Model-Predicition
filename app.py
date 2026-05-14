import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.loader import load_dataset
from utils.visualize import generate_plots
from utils.model_Predict import detect_problem_type

from utils.regression_models import train_regression_models
from utils.classification_models import train_classification_models
from utils.nlp_models import train_nlp_models

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER = os.path.join(BASE_DIR, "datasets")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# HOME ROUTE
@app.route("/", methods=["GET", "POST"])
def home():
    files = [
        file for file in os.listdir(DATASET_FOLDER)
        if file.endswith(".csv")
    ]

    table = None
    description = None
    rows, cols = None, None
    dtypes = None

    numerical_table = None
    numerical_desc = None

    categorical_table = None
    categorical_desc = None

    outlier_report = []
    correlation_report = []

    if request.method == "POST":

        df = None
        selected_file = None
        input_file = None
        if "upload_csv" in request.form:
            input_file = request.files.get("fileInput")
            if (
                input_file
                and input_file.filename
                and input_file.filename.endswith(".csv")
            ):
                filename = secure_filename(
                    input_file.filename
                )

                save_path = os.path.join(
                    DATASET_FOLDER,
                    filename
                )

                input_file.save(save_path)
                print("FILE SAVED:", filename)
                files = os.listdir(DATASET_FOLDER)
                df = pd.read_csv(save_path)

        elif "load_dataset" in request.form:
            selected_file = request.form.get(
                "dataset"
            )
            if selected_file:
                print("SELECTED:", selected_file)
                dataset_path = os.path.join(
                    DATASET_FOLDER,
                    selected_file
                )
                if os.path.exists(dataset_path):

                    df = pd.read_csv(dataset_path)

        if df is not None:
            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Remove completely empty rows
            df.dropna(how="all", inplace=True)

            # Fill numeric missing values
            numeric_cols = df.select_dtypes(
                include=["number"]
            ).columns

            for col in numeric_cols:
                df[col] = df[col].fillna(
                    df[col].mean()
                )

            # Fill categorical missing values
            categorical_cols = df.select_dtypes(
                include=["object"]
            ).columns

            for col in categorical_cols:
                df[col] = df[col].fillna(
                    "Unknown"
                )

            if input_file:
                cleaned_filename = (
                    f"cleaned_{secure_filename(input_file.filename)}"
                )
            elif selected_file:
                cleaned_filename = (
                    f"cleaned_{selected_file}"
                )
            else:
                cleaned_filename = (
                    "cleaned_data.csv"
                )

            cleaned_output_path = os.path.join(
                OUTPUT_FOLDER,
                cleaned_filename
            )

            df.to_csv(
                cleaned_output_path,
                index=False
            )

            print(
                "CLEANED DATA SAVED:",
                cleaned_filename
            )

            table = df.head().to_html(
                classes="table table-bordered"
            )

            try:
                description = df.describe(
                    include="all"
                ).to_html(classes="table")
            except Exception as e:
                description = (
                    f"<p>Error generating description: {e}</p>"
                )

            rows, cols = df.shape
            dtypes = (
                df.dtypes
                .to_frame(name="Data Type")
                .to_html(
                    classes="table table-bordered"
                )
            )

            categorical_df = df.select_dtypes(
                include=["object"]
            )

            if not categorical_df.empty:
                categorical_table = (
                    categorical_df.head().to_html(
                        classes="table table-bordered"
                    )
                )
                categorical_desc = (
                    categorical_df.describe().to_html(
                        classes="table table-bordered"
                    )
                )

            numerical_df = df.select_dtypes(
                include=["number"]
            )

            if not numerical_df.empty:
                numerical_table = (
                    numerical_df.head().to_html(
                        classes="table table-bordered"
                    )
                )
                numerical_desc = (
                    numerical_df.describe().to_html(
                        classes="table table-bordered"
                    )
                )

                for column in numerical_df.columns:
                    Q1 = numerical_df[column].quantile(0.25)
                    Q3 = numerical_df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = numerical_df[column][
                        (numerical_df[column] < lower)
                        | (numerical_df[column] > upper)
                    ]
                    outlier_count = len(outliers)
                    if outlier_count > 0:
                        percent = (
                            outlier_count / len(df)
                        ) * 100

                        outlier_report.append(
                            f"Column '{column}' contains "
                            f"{outlier_count} outliers "
                            f"({percent:.1f}% of data)."
                        )

                corr_matrix = numerical_df.corr()
                columns = corr_matrix.columns
                for i in range(len(columns)):
                    for j in range(i + 1, len(columns)):
                        col1 = columns[i]
                        col2 = columns[j]
                        corr_value = corr_matrix.loc[
                            col1,
                            col2
                        ]
                        if corr_value > 0.7:
                            correlation_report.append(
                                f"{col1} strongly affects "
                                f"{col2} "
                                f"(Correlation: {corr_value:.2f})"
                            )
                        elif corr_value < -0.7:
                            correlation_report.append(
                                f"{col1} strongly negatively affects "
                                f"{col2} "
                                f"(Correlation: {corr_value:.2f})"
                            )

    return render_template(
        "index.html",
        files=files,
        table=table,
        description=description,
        rows=rows,
        cols=cols,
        dtypes=dtypes,
        numerical_table=numerical_table,
        numerical_desc=numerical_desc,
        categorical_table=categorical_table,
        categorical_desc=categorical_desc,
        outlier_report=outlier_report,
        correlation_report=correlation_report,
    )

# VISUALIZATION ROUTE
@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    files = [
        file for file in os.listdir(OUTPUT_FOLDER)
        if file.endswith(".csv")
    ]
    plot_paths = []
    if request.method == "POST":
        selected_file = request.form.get("dataset")
        if selected_file:
            path = os.path.join(
                OUTPUT_FOLDER,
                selected_file
            )
            if os.path.exists(path):
                df = pd.read_csv(path)
                plot_paths = generate_plots(df)

    return render_template(
        "visualize.html",
        files=files,
        plot_paths=plot_paths
    )

# ANALYSIS ROUTE
@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    result = None
    predictions = None
    table = None
    columns = []
    selected_file = None
    target_column = None

    files = [
        file for file in os.listdir(OUTPUT_FOLDER)
        if file.endswith(".csv")
    ]

    if request.method == "POST":
        selected_file = request.form.get("dataset")
        target_column = request.form.get(
            "target_column",
            None
        )

        if selected_file:
            path = os.path.join(
                OUTPUT_FOLDER,
                selected_file
            )
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Clean column names
                df.columns = (
                    df.columns
                    .str.strip()
                    .str.lower()
                )

                # Clean target column
                if target_column:

                    target_column = (
                        target_column
                        .strip()
                        .lower()
                    )

                table = df.head().to_html(
                    classes="table table-bordered"
                )

                columns = df.columns.tolist()
                result = detect_problem_type(
                    df,
                    target_column
                )

                problem_type = result["problem_type"]
                if target_column:
                    if problem_type == "Regression":
                        predictions = (
                            train_regression_models(
                                df,
                                target_column,
                                result
                            )
                        )
                    elif problem_type == "Classification":
                        predictions = (
                            train_classification_models(
                                df,
                                target_column,
                                result
                            )
                        )
                    elif problem_type == "NLP Classification":

                        predictions = (
                            train_nlp_models(
                                df,
                                target_column,
                                result
                            )
                        )

    return render_template(
        "analysis.html",

        files=files,
        table=table,
        columns=columns,

        selected_file=selected_file,
        target_column=target_column,

        result=result,

        predictions=(
            predictions["all_results"]
            if predictions else None
        ),

        best_model=(
            predictions["best_model"]
            if predictions else None
        )
    )

# MAIN
if __name__ == "__main__":
    app.run()