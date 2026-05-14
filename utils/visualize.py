import os
import uuid
import matplotlib

# IMPORTANT for Render deployment
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots(df):
    plt.clf()
    plt.close("all")

    plot_paths = []

    plot_folder = os.path.join(
        "static",
        "plots"
    )

    # Create folder if missing
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(plot_folder):
        file_path = os.path.join(
            plot_folder,
            file
        )
        if os.path.isfile(file_path):
            os.remove(file_path)

    if len(df) > 1000:
        df = df.sample(
            1000,
            random_state=42
        )

    # Numerical Columns
    numerical_df = df.select_dtypes(
        include=["number"]
    )

    # Limit columns
    numerical_columns = numerical_df.columns[:5]
    for col in numerical_columns:
        try:
            plt.figure(figsize=(6, 4))
            if numerical_df[col].nunique() > 1:
                sns.histplot(
                    numerical_df[col].dropna(),
                    kde=True
                )
            else:
                sns.histplot(
                    numerical_df[col].dropna()
                )

            plt.title(f"{col} Distribution")
            hist_filename = (
                f"{uuid.uuid4()}_hist.png"
            )
            hist_path = os.path.join(
                plot_folder,
                hist_filename
            )
            plt.savefig(
                hist_path,
                bbox_inches="tight",
                dpi=70
            )
            plt.close()
            plot_paths.append(
                f"plots/{hist_filename}"
            )

            # Boxplot
            plt.figure(figsize=(6, 2))
            sns.boxplot(
                x=numerical_df[col].dropna()
            )
            plt.title(f"{col} Boxplot")
            box_filename = (
                f"{uuid.uuid4()}_box.png"
            )
            box_path = os.path.join(
                plot_folder,
                box_filename
            )
            plt.savefig(
                box_path,
                bbox_inches="tight",
                dpi=70
            )
            plt.close()
            plot_paths.append(
                f"plots/{box_filename}"
            )

        except Exception as e:
            print(
                f"Error generating numerical plot for {col}:",
                e
            )
            plt.close("all")

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    categorical_columns = categorical_df.columns[:5]

    for col in categorical_columns:
        try:
            unique_count = (
                categorical_df[col].nunique()
            )

            if unique_count <= 10:
                top_values = (
                    categorical_df[col]
                    .value_counts()
                    .head(10)
                )
                # Countplot
                plt.figure(figsize=(7, 4))
                sns.countplot(
                    x=categorical_df[col],
                    order=top_values.index
                )
                plt.xticks(rotation=45)
                plt.title(f"{col} Countplot")
                count_filename = (
                    f"{uuid.uuid4()}_count.png"
                )
                count_path = os.path.join(
                    plot_folder,
                    count_filename
                )
                plt.savefig(
                    count_path,
                    bbox_inches="tight",
                    dpi=70
                )
                plt.close()
                plot_paths.append(
                    f"plots/{count_filename}"
                )

                # Pie Chart
                plt.figure(figsize=(5, 5))
                top_values.plot(
                    kind="pie",
                    autopct="%1.1f%%"
                )
                plt.ylabel("")
                plt.title(f"{col} Pie Chart")
                pie_filename = (
                    f"{uuid.uuid4()}_pie.png"
                )
                pie_path = os.path.join(
                    plot_folder,
                    pie_filename
                )
                plt.savefig(
                    pie_path,
                    bbox_inches="tight",
                    dpi=70
                )
                plt.close()
                plot_paths.append(
                    f"plots/{pie_filename}"
                )

            elif unique_count <= 30:
                top_categories = (
                    categorical_df[col]
                    .value_counts()
                    .head(10)
                )
                plt.figure(figsize=(8, 4))
                sns.barplot(
                    x=top_categories.index,
                    y=top_categories.values
                )
                plt.xticks(rotation=45)
                plt.title(f"Top 10 {col}")
                top_filename = (
                    f"{uuid.uuid4()}_top10.png"
                )
                top_path = os.path.join(
                    plot_folder,
                    top_filename
                )
                plt.savefig(
                    top_path,
                    bbox_inches="tight",
                    dpi=70
                )
                plt.close()
                plot_paths.append(
                    f"plots/{top_filename}"
                )
        except Exception as e:
            print(
                f"Error generating categorical plot for {col}:",
                e
            )
            plt.close("all")

    # Final Cleanup
    plt.clf()
    plt.close("all")

    return plot_paths