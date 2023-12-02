import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from correction.cluster_of_words.evaluator import ClusterOfWordsEvaluator


class PlotterForClusterOfWords:
    grade_column = "second_grade"
    perplexity_column = "perplexity"
    colors = ["red", "green", "purple", "orange", "black", "brown"]

    def plot_perplexity_for_each_essay(self, df: pd.DataFrame, title="Perplexity"):
        x = df.index
        y = df[self.perplexity_column]

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(x, y, label="Perplexity")

        last_ocurrence_of_each_grade = self._get_last_occurence_of_each_grade(df)

        for i, idx in enumerate(last_ocurrence_of_each_grade):
            ax.axvline(
                x=idx,
                color=self.colors[i],
                linestyle="--",
                label=df.loc[idx, self.grade_column],
            )

        ax.grid(True, linestyle="--", color="gray", alpha=0.6)

        ax.set_xlabel("Essay count")
        ax.set_ylabel(title)
        ax.set_title("Perplexity of essays")
        ax.legend()

        lower_limit = min(y)
        upper_limit = max(y)
        ax.set_ylim(lower_limit, upper_limit)
        plt.show()

    def _get_last_occurence_of_each_grade(self, df: pd.DataFrame):
        possible_grades = df[self.grade_column].unique().tolist()
        last_ocurrence_of_each_grade = []

        for grade in possible_grades:
            last_ocurrence_index = df[df[self.grade_column] == grade].index.tolist()[-1]
            last_ocurrence_of_each_grade.append(last_ocurrence_index)

        return last_ocurrence_of_each_grade

    def plot_perplexity_metric_for_each_grade(self, metric_df, bar_color, title):
        ax = metric_df.plot(
            x=self.grade_column,
            y="perplexity",
            kind="bar",
            ylabel="perplexity",
            grid=True,
            color=bar_color,
            title=title,
        )
        ax.yaxis.grid(color="grey", linestyle="--", linewidth=0.5)
        ax.xaxis.grid(color="grey", linestyle="--", linewidth=0.5)
        plt.show()

    def plot_perplexity_mean_for_each_grade(self, mean_df: pd.DataFrame):
        self.plot_perplexity_metric_for_each_grade(
            metric_df=mean_df, bar_color="blue", title="Perplexity mean by each grade"
        )

    def plot_perplexity_std_dev_for_each_grade(self, std_dev_df: pd.DataFrame):
        self.plot_perplexity_metric_for_each_grade(
            metric_df=std_dev_df, bar_color="green", title="Perplexity std-dev by each grade"
        )

    def plot_confusion_matrix_for_confidence_intervals(
        self, perplexity_df: pd.DataFrame, confidence_intervals: dict
    ):
        for confidence, interval in confidence_intervals.items():
            df = perplexity_df.copy()

            lower_bound, upper_bound = interval

            evaluator = ClusterOfWordsEvaluator(lower_bound)

            df["true_label"] = df["second_grade"] == 40
            df["predicted_label"] = df["perplexity"].apply(evaluator.evaluate)

            matrix = confusion_matrix(df["true_label"], df["predicted_label"])

            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.8)
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

            plt.xlabel("Predicted")
            plt.ylabel("Aglomerado de palavras")
            plt.title(f"Matriz de confusão - Confiança {confidence*100}%")

            class_names = ["Negative", "Positive"]
            tick_marks = [0.5, 1.5]
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)

            plt.show()
