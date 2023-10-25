from typing import Callable
import scipy.stats as stats

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    BertLMHeadModel,
)
import torch
from tqdm.auto import tqdm


class PerplexityCalculator:
    text_column = "essay_text"
    perplexity_column = "perplexity"
    grade_column = "second_grade"
    id_column = "essay_id"

    def __init__(self, checkpoint: str):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = BertLMHeadModel.from_pretrained(checkpoint, is_decoder=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def calculate_text_perplexity(self, text: str):
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            log_likelihood = outputs.loss.item()

        perplexity = np.exp(-log_likelihood)

        return perplexity

    def calculate_perplexity_for_all_essays(self, df: pd.DataFrame):
        perplexities = []
        progress_bar = tqdm(range(len(df.index)))

        for essay in df[self.text_column].tolist():
            p = self.calculate_text_perplexity(essay)
            perplexities.append(p)
            progress_bar.update(1)

        df[self.perplexity_column] = perplexities
        df = df.sort_values(by=[self.grade_column, self.perplexity_column]).reset_index(drop=True)
        return df

    def calculate_perplexity_mean_for_each_grade(self, df: pd.DataFrame):
        return self._calculate_perplexity_metric_for_each_grade(
            df, lambda group_by: group_by.mean()
        )

    def calculate_perplexity_std_dev_for_each_grade(self, df: pd.DataFrame):
        return self._calculate_perplexity_metric_for_each_grade(df, lambda group_by: group_by.std())

    def _calculate_perplexity_metric_for_each_grade(self, df: pd.DataFrame, agg_function: Callable):
        metric_df = df[[self.grade_column, self.perplexity_column]].groupby(self.grade_column)
        metric_df = agg_function(metric_df).reset_index(names=self.grade_column)
        return metric_df

    def calculate_perplexity_for_all_essays_using_chunks_mean(self, df: pd.DataFrame):
        mean_df = self._calculate_perplexity_for_all_essays_aggregating_chunks(
            df, lambda group_by: group_by.mean()
        )
        return mean_df

    def calculate_perplexity_for_all_essays_using_chunks_sum(self, df: pd.DataFrame):
        sum_df = self._calculate_perplexity_for_all_essays_aggregating_chunks(
            df, lambda group_by: group_by.sum()
        )
        return sum_df

    def _calculate_perplexity_for_all_essays_aggregating_chunks(
        self, chunks_df: pd.DataFrame, agg_function: Callable
    ):
        metric_df = chunks_df[[self.id_column, self.perplexity_column]].groupby(self.id_column)
        metric_df = agg_function(metric_df).reset_index()
        metric_df = (
            metric_df.join(
                chunks_df.set_index(self.id_column).drop(columns=self.perplexity_column),
                on=self.id_column,
            )
            .sort_values(by=[self.grade_column, self.perplexity_column])
            .reset_index(drop=True)
        )
        return metric_df

    def calculate_confidence_intervals(self, df: pd.DataFrame, confidence_levels: list):
        perplexity_mean = df["perplexity"].mean()
        perplexity_sem = stats.sem(df["perplexity"])  # standard error of the mean

        sample_size = len(df.index)

        confidence_to_interval = {}

        for confidence_level in confidence_levels:
            t_score = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
            margin_of_error = t_score * perplexity_sem
            lower_bound = perplexity_mean - margin_of_error
            upper_bound = perplexity_mean + margin_of_error
            confidence_to_interval[confidence_level] = (lower_bound, upper_bound)
            print(
                f"{confidence_level * 100}% confidence interval: ({lower_bound:.10f}, {upper_bound:.10f})"
            )

        return confidence_to_interval
