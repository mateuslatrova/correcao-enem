from datasets import Dataset, DatasetDict

class EssayDatasetPreprocessorForClusterOfWords:
    RENAME_COLUMNS_MAP = {
        "id": "essay_id",
        "essay": "essay_text",
    }
    SECOND_GRADE_INDEX = 1
    GRADE_THRESHOLD = 0
    TEST_DATASET_SIZE = 0.15
    ESSAY_ID_COLUMN = "essay_id"

    def __init__(self, dataset: Dataset, columns_to_remove: list):
        self.original_dataset = dataset
        self.columns_to_remove = columns_to_remove

    def preprocess_dataset(self) -> DatasetDict:
        self.train_dataset = (
            self.original_dataset.rename_columns(self.RENAME_COLUMNS_MAP)
            .map(self._preprocess_essay)
            .map(self._create_column_for_second_grade)
        )

        self.train_dataset = self.train_dataset.remove_columns(self.columns_to_remove)

        self.dataset_dict = DatasetDict({"train": self.train_dataset})

        return self.dataset_dict

    def _preprocess_essay(self, example):
        essay_before = example["essay_text"]
        essay_after = ""

        paragraphs = eval(essay_before)
        for paragraph in paragraphs:
            essay_after += paragraph

        example["essay_text"] = essay_after
        return example

    def _create_column_for_second_grade(self, example):
        grades_list = eval(example["grades"])
        second_grade = grades_list[self.SECOND_GRADE_INDEX]
        example["second_grade"] = second_grade
        return example