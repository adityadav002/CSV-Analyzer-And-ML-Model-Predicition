import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df):
    plot_paths = []
    plot_folder = "static/plots"

    # Remove old plots
    for file in os.listdir(plot_folder):

        file_path = os.path.join(plot_folder, file)

        if os.path.isfile(file_path):
            os.remove(file_path)

    # =========================
    # Numerical Columns
    # =========================

    numerical_df = df.select_dtypes(include=["number"])

    for col in numerical_df.columns:
        # Histogram + KDE
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
        hist_path = f"plots/{col}_hist.png"
        plt.savefig(
            f"static/{hist_path}",
            bbox_inches="tight"
        )
        plt.close()
        plot_paths.append(hist_path)

        # Boxplot
        plt.figure(figsize=(6, 2))
        sns.boxplot(
            x=numerical_df[col].dropna()
        )
        plt.title(f"{col} Boxplot")
        box_path = f"plots/{col}_box.png"
        plt.savefig(
            f"static/{box_path}",
            bbox_inches="tight"
        )
        plt.close()
        plot_paths.append(box_path)

    # =========================
    # Categorical Columns
    # =========================

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    for col in categorical_df.columns:
        unique_count = categorical_df[col].nunique()
        # Small category columns
        if unique_count <= 10:
            # Countplot
            plt.figure(figsize=(7, 4))
            sns.countplot(
                x=categorical_df[col],
                order=categorical_df[col]
                .value_counts()
                .index
            )
            plt.xticks(rotation=45)
            plt.title(f"{col} Countplot")
            count_path = f"plots/{col}_count.png"
            plt.savefig(
                f"static/{count_path}",
                bbox_inches="tight"
            )
            plt.close()
            plot_paths.append(count_path)

            # Pie chart
            plt.figure(figsize=(6, 6))
            categorical_df[col].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%"
            )
            plt.ylabel("")
            plt.title(f"{col} Pie Chart")
            pie_path = f"plots/{col}_pie.png"
            plt.savefig(
                f"static/{pie_path}",
                bbox_inches="tight"
            )
            plt.close()
            plot_paths.append(pie_path)

        # Medium category columns
        elif unique_count <= 30:
            plt.figure(figsize=(10, 5))
            top_categories = (
                categorical_df[col]
                .value_counts()
                .head(10)
            )
            sns.barplot(
                x=top_categories.index,
                y=top_categories.values
            )
            plt.xticks(rotation=45)
            plt.title(f"Top 10 {col}")
            top_path = f"plots/{col}_top10.png"
            plt.savefig(
                f"static/{top_path}",
                bbox_inches="tight"
            )
            plt.close()
            plot_paths.append(top_path)

    return plot_paths