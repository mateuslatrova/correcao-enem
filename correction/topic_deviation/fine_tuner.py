from collections import defaultdict

from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class TopicDeviationFineTuner:
    columns_to_remove_after_tokenization = ["essay_id", "essay_text", "topic_text", "topic_id"]
    datasets_format = "torch"
    original_label_column = "label"
    new_label_column = "labels"
    splits = ["test", "train", "validation"]

    def __init__(
        self, checkpoint: str, datasets: DatasetDict, learning_rate: float = 5e-5, batch_size=8
    ) -> None:
        self.checkpoint = checkpoint
        self.datasets = datasets
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, do_lower_case=False)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.tokenized_datasets = self.get_tokenized_datasets()
        self.data_loaders = self.get_data_loaders()
        self.metrics = defaultdict(list)

    def get_tokenized_datasets(self):
        tokenized_datasets = self.datasets.map(self.tokenize, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(
            self.columns_to_remove_after_tokenization
        )
        tokenized_datasets = tokenized_datasets.rename_column(
            self.original_label_column, self.new_label_column
        )

        tokenized_datasets.set_format(self.datasets_format)
        return tokenized_datasets

    def tokenize(self, example):
        return self.tokenizer(
            example["essay_text"],
            example["topic_text"],
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def get_data_loaders(self):
        data_loaders = {}

        for split in self.splits:
            data_loaders[split] = DataLoader(
                self.tokenized_datasets[split],
                shuffle=True,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
            )

        return data_loaders

    def run_model_training(self, num_epochs: int = 3):
        train_dataloader = self.data_loaders["train"]

        num_training_steps = num_epochs * len(train_dataloader)

        self.learning_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        device = self.get_current_device()
        self.set_model_device(device)

        self.train_progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_epochs):
            # Training loop:
            self.model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                self.train_model_for_batch(batch)

            # Validation loop:
            self.run_model_evaluation("validation")

    def set_model_device(self, device):
        self.model.to(device)

    def get_current_device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train_model_for_batch(self, batch):
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()

        self.optimizer.step()
        self.learning_scheduler.step()
        self.optimizer.zero_grad()
        self.train_progress_bar.update(1)

    def run_model_evaluation(self, split: str):
        dataloader = self.data_loaders[split]

        device = self.get_current_device()
        self.set_model_device(device)

        self.model.eval()
        metric = evaluate.load("glue", "mrpc")
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch[self.new_label_column])

        self.metrics[split].append(metric.compute())

    def run_model_test(self):
        self.run_model_evaluation("test")
