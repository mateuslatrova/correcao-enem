from collections import defaultdict

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding, get_scheduler

from correction.topic_deviation.neural_network import TopicDeviationNeuralNetwork


class TopicDeviationFineTuner:
    columns_to_remove_after_tokenization = [
        "essay_id",
        "essay_text",
        "topic_text",
        "topic_id",
        "label",
    ]
    datasets_format = "torch"
    original_label_column = "label"
    new_label_column = "labels"
    splits = ["test", "train", "validation"]

    def __init__(
        self, checkpoint: str, datasets: DatasetDict, learning_rate: float = 5e-5, batch_size=4
    ) -> None:
        self.checkpoint = checkpoint
        self.datasets = datasets
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, do_lower_case=False)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.neural_network = TopicDeviationNeuralNetwork(self.checkpoint)
        self.optimizer = AdamW(self.neural_network.parameters(), lr=learning_rate)

        self.tokenized_essay_datasets = self.get_essay_tokenized_datasets()
        self.tokenized_topic_datasets = self.get_topic_tokenized_datasets()

        self.metrics = defaultdict(list)

        self.split_to_results_by_epoch = {split: {} for split in self.splits}

    def get_essay_tokenized_datasets(self):
        tokenized_essay_datasets = self.datasets.map(self.tokenize_essay, batched=True)

        tokenized_essay_datasets = tokenized_essay_datasets.remove_columns(
            self.columns_to_remove_after_tokenization
        )

        tokenized_essay_datasets.set_format(self.datasets_format)
        return tokenized_essay_datasets

    def get_topic_tokenized_datasets(self):
        tokenized_topic_datasets = self.datasets.map(self.tokenize_topic, batched=True)

        tokenized_topic_datasets = tokenized_topic_datasets.remove_columns(
            self.columns_to_remove_after_tokenization
        )

        tokenized_topic_datasets.set_format(self.datasets_format)
        return tokenized_topic_datasets

    def tokenize_essay(self, example):
        return self.tokenizer(
            example["essay_text"],
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def tokenize_topic(self, example):
        return self.tokenizer(
            example["topic_text"],
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def get_data_loaders(self, datasets: DatasetDict):
        data_loaders = {}

        for split in self.splits:
            data_loaders[split] = DataLoader(
                datasets[split],
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
            )

        return data_loaders

    def get_label_data_loader(self, datasets: DatasetDict):
        data_loaders = {}
        labels_datasets = DatasetDict()

        for split in self.splits:
            split_data = datasets[split]
            label_column = split_data[self.original_label_column]

            new_dataset = Dataset.from_dict({self.original_label_column: label_column})

            labels_datasets[split] = new_dataset

        labels_datasets = labels_datasets.rename_column(
            self.original_label_column, self.new_label_column
        )

        for split in self.splits:
            data_loaders[split] = DataLoader(
                labels_datasets[split],
                batch_size=self.batch_size,
            )

        return data_loaders

    def run_model_training(self, num_epochs: int = 3):
        essay_train_dataloader = self.get_data_loaders(self.tokenized_essay_datasets)["train"]
        topic_train_dataloader = self.get_data_loaders(self.tokenized_topic_datasets)["train"]
        label_train_dataloader = self.get_label_data_loader(self.datasets)["train"]

        num_training_steps = num_epochs * len(essay_train_dataloader)

        self.learning_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        device = self.get_current_device()
        self.set_neural_network_device(device)

        self.train_progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_epochs):
            self.neural_network.train()

            essays_iterator = iter(essay_train_dataloader)
            topics_iterator = iter(topic_train_dataloader)
            labels_iterator = iter(label_train_dataloader)

            dataloaders = zip(essays_iterator, topics_iterator, labels_iterator)

            accumulation_steps = 4

            for essay_batch, topic_batch, label_batch in dataloaders:
                essay_batch = {k: v.to(device) for k, v in essay_batch.items()}
                topic_batch = {k: v.to(device) for k, v in topic_batch.items()}
                label_batch = {k: v.to(device) for k, v in label_batch.items()}

                outputs = self.neural_network(essay_batch, topic_batch, label_batch)

                loss = outputs["loss"]
                loss.backward()

                # Perform optimization step after accumulation_steps batches
                if self.train_progress_bar.n % accumulation_steps == 0:
                    self.optimizer.step()
                    self.learning_scheduler.step()
                    self.optimizer.zero_grad()

                self.train_progress_bar.update(1)

            self.run_model_evaluation("validation", epoch)
            self.run_model_evaluation("test", epoch)

    def set_neural_network_device(self, device):
        self.neural_network.to(device)

    def get_current_device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def run_model_evaluation(self, split: str, epoch=None):
        device = self.get_current_device()
        self.set_neural_network_device(device)

        self.neural_network.eval()
        metric = evaluate.load("glue", "mrpc")

        essay_train_dataloader = self.get_data_loaders(self.tokenized_essay_datasets)[split]
        topic_train_dataloader = self.get_data_loaders(self.tokenized_topic_datasets)[split]
        label_train_dataloader = self.get_label_data_loader(self.datasets)[split]

        essays_iterator = iter(essay_train_dataloader)
        topics_iterator = iter(topic_train_dataloader)
        labels_iterator = iter(label_train_dataloader)

        dataloaders = zip(essays_iterator, topics_iterator, labels_iterator)

        true_labels = []
        predicted_labels = []

        for essay_batch, topic_batch, label_batch in dataloaders:
            essay_batch = {k: v.to(device) for k, v in essay_batch.items()}
            topic_batch = {k: v.to(device) for k, v in topic_batch.items()}

            with torch.no_grad():
                outputs = self.neural_network(essay_batch, topic_batch)

            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)

            metric.add_batch(predictions=predictions, references=label_batch["labels"])

            true_labels.extend(label_batch["labels"].tolist())
            predicted_labels.extend(predictions.cpu().tolist())

        self.metrics[split].append(metric.compute())
        if epoch is not None:
            self.metrics[split][-1]["epoch"] = epoch

        df = pd.DataFrame({"true_label": true_labels, "predicted_label": predicted_labels})

        self.split_to_results_by_epoch[split][epoch] = df

    def run_model_test(self):
        self.run_model_evaluation("test")

    def show_metrics(self):
        for split, metrics in self.metrics.items():
            print(split)
            for metric in metrics:
                print(metric)

    def plot_confusion_matrices(self, split):
        for epoch, df in self.split_to_results_by_epoch[split].items():
            title = f"Results for split {split} in epoch {epoch}"
            self.plot_confusion_matrix(title, df)

    def plot_confusion_matrix(self, title, df):
        matrix = confusion_matrix(df["true_label"], df["predicted_label"])

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.8)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

        plt.xlabel("Predicted")
        plt.ylabel("Fugiu ao tema")
        plt.title(title)

        class_names = ["Negative", "Positive"]
        tick_marks = [0.5, 1.5]
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        plt.show()
