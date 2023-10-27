from datasets import Dataset


class PromptDatasetPreprocessorForTopicDeviation:
    COLUMNS_TO_REMOVE = ["source", "title", "prompt", "supporting_text"]
    RENAME_COLUMNS_MAP = {
        "id": "topic_id",
    }
    PARAGRAPH_ENDINGS = [".", "?", "!"]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def preprocess_dataset(self):
        self.preprocessed_dataset = (
            self.dataset.rename_columns(self.RENAME_COLUMNS_MAP)
            .map(self._concatenate_texts)
            .remove_columns(self.COLUMNS_TO_REMOVE)
        )
        return self.preprocessed_dataset

    def _concatenate_texts(self, example):
        concatenated_texts = ""

        topic = example["prompt"]
        supporting_text = example["supporting_text"]

        # This try-catch block is necessary because some titles have quotes around it,
        # but others don't.
        try:
            title = eval(example["title"])
        except SyntaxError:
            title = example["title"]

        paragraphs = [title] + eval(supporting_text) + eval(topic)

        for paragraph in paragraphs:
            has_desired_ending = False
            for ending in self.PARAGRAPH_ENDINGS:
                if paragraph.endswith(ending):
                    has_desired_ending = True
                    break

            if not has_desired_ending:
                paragraph = paragraph + "."

            concatenated_texts += paragraph

        example["topic_text"] = concatenated_texts
        return example
