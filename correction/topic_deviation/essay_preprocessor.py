import math
import random

from datasets import Dataset, DatasetDict, concatenate_datasets


class EssayDatasetPreprocessorForTopicDeviation:
    COLUMNS_TO_REMOVE = [
        "title",
        "final_grade",
        "is_ENEM",
        "is_convertible",
        "general",
        "specific",
        "grades",
    ]
    RENAME_COLUMNS_MAP = {
        "id": "essay_id",
        "id_prompt": "topic_id",
        "essay": "essay_text",
    }

    SECOND_GRADE_INDEX = 1
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    GRADE_THRESHOLD = 0

    TEST_DATASET_SIZE = 0.15
    VALIDATION_DATASET_SIZE = 0.15
    ORIGINAL_NEGATIVE_EXAMPLES_IN_VALIDATION = 3

    ESSAY_ID_COLUMN = "essay_id"
    IS_ARTIFICIAL_COLUMN = "is_artificial"

    def __init__(self, dataset: Dataset):
        self.original_dataset = dataset

    def preprocess_dataset(self) -> DatasetDict:
        self.train_dataset = (
            self.original_dataset.rename_columns(self.RENAME_COLUMNS_MAP)
            .map(self._preprocess_essay)
            .map(self._create_column_for_topic_deviation_label)
            .map(lambda example: self._set_is_artificial_value(False, example))
        )
        self.train_dataset = self._create_column_to_categorize_artificial_examples(
            self.train_dataset, is_artificial=False
        )

        self.artificial_negative_examples_dataset = (
            self._create_artificial_negative_examples_dataset()
        )
        self.artificial_negative_examples_dataset = (
            self._create_column_to_categorize_artificial_examples(
                self.artificial_negative_examples_dataset, is_artificial=True
            )
        )

        self.train_dataset = concatenate_datasets(
            [self.train_dataset, self.artificial_negative_examples_dataset]
        ).remove_columns(self.COLUMNS_TO_REMOVE)

        self.train_dataset_original_size = len(self.train_dataset)

        self.test_dataset = self._create_test_dataset()

        self.train_dataset = self._remove_rows_from_dataset(
            rows_to_remove=self.test_dataset, dataset_to_remove_from=self.train_dataset
        )

        self.validation_dataset = self._create_validation_dataset()

        self.train_dataset = self._remove_rows_from_dataset(
            rows_to_remove=self.validation_dataset, dataset_to_remove_from=self.train_dataset
        )

        self.dataset_dict = DatasetDict(
            {
                "train": self.train_dataset,
                "test": self.test_dataset,
                "validation": self.validation_dataset,
            }
        )

        return self.dataset_dict

    def _preprocess_essay(self, example):
        essay_before = example["essay_text"]
        essay_after = ""

        paragraphs = eval(essay_before)
        for paragraph in paragraphs:
            essay_after += paragraph

        example["essay_text"] = essay_after
        return example

    def _create_column_for_topic_deviation_label(self, example):
        grades_list = eval(example["grades"])
        second_grade = grades_list[self.SECOND_GRADE_INDEX]

        if second_grade > self.GRADE_THRESHOLD:
            example["label"] = self.POSITIVE_LABEL
        else:
            example["label"] = self.NEGATIVE_LABEL

        return example

    def _create_column_to_categorize_artificial_examples(self, dataset, is_artificial):
        return dataset.map(lambda example: self._set_is_artificial_value(is_artificial, example))

    def _set_is_artificial_value(self, is_artificial, example):
        example["is_artificial"] = is_artificial
        return example

    def _create_artificial_negative_examples_dataset(self):
        positive_examples_dataset = self.train_dataset.filter(
            lambda example: example["label"] == self.POSITIVE_LABEL
        )
        negative_examples_dataset = self.train_dataset.filter(
            lambda example: example["label"] == self.NEGATIVE_LABEL
        )

        positive_examples_dataset.set_format("pandas")

        positive_examples_df = positive_examples_dataset[:]

        number_of_artificial_examples = len(positive_examples_dataset) - len(
            negative_examples_dataset
        )

        artificial_negative_examples_df = positive_examples_df.sample(
            n=number_of_artificial_examples
        )

        min_topic_id = min(self.train_dataset["topic_id"])
        max_topic_id = max(self.train_dataset["topic_id"])

        artificial_negative_examples_df["topic_id"] = artificial_negative_examples_df[
            "topic_id"
        ].apply(
            lambda topic_id: self._random_integer_with_blacklist(
                min_topic_id, max_topic_id, [topic_id]
            )
        )
        artificial_negative_examples_df["label"] = self.NEGATIVE_LABEL

        artificial_negative_examples_dataset = Dataset.from_pandas(
            artificial_negative_examples_df, preserve_index=False
        )

        return artificial_negative_examples_dataset

    def _random_integer_with_blacklist(self, min_val, max_val, blacklist):
        while True:
            random_num = random.randint(min_val, max_val)
            if random_num not in blacklist:
                return random_num

    def _create_test_dataset(self):
        original_negative_examples = self.train_dataset.filter(
            lambda example: example["label"] == self.NEGATIVE_LABEL
            and example["is_artificial"] == False
        )
        original_positive_examples = self.train_dataset.filter(
            lambda example: example["label"] == self.POSITIVE_LABEL
            and example["is_artificial"] == False
        )

        num_of_examples = int(self.train_dataset_original_size * self.TEST_DATASET_SIZE)

        num_of_negative_examples = int(math.floor(num_of_examples / 2))
        num_of_positive_examples = num_of_negative_examples

        test_negative_examples = original_negative_examples.shuffle().select(
            range(num_of_negative_examples)
        )

        test_positive_examples = original_positive_examples.shuffle().select(
            range(num_of_positive_examples)
        )

        test_dataset = concatenate_datasets(
            [
                test_positive_examples,
                test_negative_examples,
            ]
        ).shuffle()

        return test_dataset

    def _create_validation_dataset(self):
        original_negative_examples = self.train_dataset.filter(
            lambda example: example["label"] == self.NEGATIVE_LABEL
            and example["is_artificial"] == False
        )

        artificial_negative_examples = self.train_dataset.filter(
            lambda example: example["label"] == self.NEGATIVE_LABEL
            and example["is_artificial"] == True
        )

        positive_examples = self.train_dataset.filter(
            lambda example: example["label"] == self.POSITIVE_LABEL
        )

        num_of_examples = int(
            math.ceil(self.train_dataset_original_size * self.VALIDATION_DATASET_SIZE)
        )

        num_of_positive_examples = int(math.ceil(num_of_examples / 2))
        num_of_negative_examples = int(math.floor(num_of_examples / 2))

        positive_examples_essay_ids = []
        for elem in positive_examples.shuffle():
            if len(positive_examples_essay_ids) == num_of_positive_examples:
                break
            positive_examples_essay_ids.append(elem["essay_id"])

        validation_positive_examples = positive_examples.filter(
            lambda example: example["essay_id"] in positive_examples_essay_ids
        )

        original_negative_examples_essay_ids = []
        for elem in original_negative_examples.shuffle():
            if (
                len(original_negative_examples_essay_ids)
                == self.ORIGINAL_NEGATIVE_EXAMPLES_IN_VALIDATION
            ):
                break
            original_negative_examples_essay_ids.append(elem["essay_id"])

        validation_original_negative_examples = original_negative_examples.filter(
            lambda example: example["essay_id"] in original_negative_examples_essay_ids
        )

        artificial_negative_examples_essay_ids = []
        for elem in artificial_negative_examples.shuffle():
            if (
                len(artificial_negative_examples_essay_ids)
                == num_of_negative_examples - self.ORIGINAL_NEGATIVE_EXAMPLES_IN_VALIDATION
            ):
                break
            artificial_negative_examples_essay_ids.append(elem["essay_id"])

        validation_artificial_negative_examples = artificial_negative_examples.filter(
            lambda example: example["essay_id"] in artificial_negative_examples_essay_ids
        )

        validation_dataset = concatenate_datasets(
            [
                validation_positive_examples,
                validation_original_negative_examples,
                validation_artificial_negative_examples,
            ]
        ).shuffle()

        return validation_dataset

    def _remove_rows_from_dataset(self, rows_to_remove: Dataset, dataset_to_remove_from: Dataset):
        ids_to_remove = list(rows_to_remove[self.ESSAY_ID_COLUMN])
        is_artificial_to_remove = list(rows_to_remove[self.IS_ARTIFICIAL_COLUMN])
        identifiers_to_remove = list(zip(ids_to_remove, is_artificial_to_remove))
        return dataset_to_remove_from.filter(
            lambda example: self._keep_example_if_not_in_remove_list(identifiers_to_remove, example)
        )

    def _keep_example_if_not_in_remove_list(self, identifiers_to_remove, example):
        essay_id = example[self.ESSAY_ID_COLUMN]
        is_artificial = example[self.IS_ARTIFICIAL_COLUMN]
        return (essay_id, is_artificial) not in identifiers_to_remove
