import os
from flask import Flask, render_template, request
import pandas as pd

from utils.loader import load_dataset
from utils.visualize import generate_plots
from utils.model_Predict import detect_problem_type

from utils.regression_models import train_regression_models
from utils.classification_models import train_classification_models
from utils.nlp_models import train_nlp_models

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    # Get all dataset names
    files = os.listdir("datasets")
    table = None
    description = None
    rows, cols = None, None
    dtypes = None
    numerical_table = None
    numerical_desc = None   
    categorical_table = None
    categorical_desc = None
    plot_paths = []
    outlier_report = []
    correlation_report = []

    # When user selects dataset
    if request.method == "POST":
        selected_file = request.form["dataset"]
        df = load_dataset(selected_file)
        table = df.head().to_html(classes="table table-bordered")

        try:
            description = df.describe(include="all").to_html(classes="table")
        except Exception as e:
            description = f"<p>Error generating description: {e}</p>"

        rows, cols = df.shape
        dtypes = df.dtypes.to_frame(name="Data Type").to_html(classes="table table-bordered")

        # Categorical columns
        categorical_df = df.select_dtypes(
            include=["object"]
        )
        if not categorical_df.empty:
            categorical_table = categorical_df.head().to_html(
                classes="table table-bordered"
            )

            categorical_desc = categorical_df.describe().to_html(
                classes="table table-bordered"
            )
            
        # Numeric columns
        numerical_df = df.select_dtypes(
            include=["number"]
        )
        if not numerical_df.empty: 
            numerical_table = numerical_df.head().to_html( classes="table table-bordered" ) 
            numerical_desc = numerical_df.describe().to_html( classes="table table-bordered" )

        for column in numerical_df.columns:
            Q1 = numerical_df[column].quantile(0.25)
            Q3 = numerical_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = numerical_df[column][
                (numerical_df[column] < lower) |
                (numerical_df[column] > upper)
            ]
            outlier_count = len(outliers)
            if outlier_count > 0:
                percent = (outlier_count / len(df)) * 100
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
                corr_value = corr_matrix.loc[col1, col2]
                if corr_value > 0.7:
                    correlation_report.append(
                        f"{col1} strongly affects {col2} "
                        f"(Correlation: {corr_value:.2f})"
                    )
                elif corr_value < -0.7:
                    correlation_report.append(
                        f"{col1} strongly negatively affects {col2} "
                        f"(Correlation: {corr_value:.2f})"
                    )

        # Generate visualizations
        plot_paths = generate_plots(df)

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
        plot_paths=plot_paths,
        outlier_report=outlier_report,
        correlation_report=correlation_report,
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    dataset_name = request.form.get("dataset")
    target_column = request.form.get("target_column")
    path = os.path.join("datasets", dataset_name)
    df = pd.read_csv(path)

    # =========================
    # DETECT PROBLEM TYPE
    # =========================
    result = detect_problem_type(
        df,
        target_column
    )

    problem_type = result["problem_type"]

    # =========================
    # TRAIN MODELS
    # =========================
    predictions = None

    if problem_type == "Regression":

        predictions = train_regression_models(
            df,
            target_column,
            result
        )

    elif problem_type == "Classification":

        predictions = train_classification_models(
            df,
            target_column,
            result
        )

    elif problem_type == "NLP Classification":

        predictions = train_nlp_models(
            df,
            target_column,
            result
        )

    return render_template(
        "result.html",
        result=result,
        predictions=predictions
    )

if __name__ == "__main__":
    app.run(debug=True)