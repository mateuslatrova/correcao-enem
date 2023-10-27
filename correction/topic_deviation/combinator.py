from datasets import Dataset, DatasetDict


class EssayAndPromptDatasetsCombinator:
    COLUMNS_ORDER = ["essay_id", "essay_text", "topic_id", "topic_text", "label"]
    SPLITS = ["test", "train", "validation"]

    def combine(self, essay_dataset_dict: DatasetDict, topic_dataset: Dataset) -> DatasetDict:
        topic_dataset.set_format("pandas")
        topic_df = topic_dataset[:].set_index("topic_id")
        combined_dataset_dict = {}

        for split in self.SPLITS:
            essay_dataset = essay_dataset_dict[split]
            essay_dataset.set_format("pandas")

            essay_df = essay_dataset[:]

            combined_df = essay_df.join(topic_df, on="topic_id")
            combined_df = combined_df.loc[:, self.COLUMNS_ORDER]

            essay_dataset.set_format()

            combined_dataset = Dataset.from_pandas(combined_df, preserve_index=False)

            combined_dataset_dict[split] = combined_dataset

        combined_dataset_dict = DatasetDict(combined_dataset_dict)

        return combined_dataset_dict
