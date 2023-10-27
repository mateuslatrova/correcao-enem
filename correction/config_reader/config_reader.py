import os
import yaml


class ConfigReader:
    data_dirname = "data"
    raw_files_dirname = "raw"
    preprocessed_files_dirname = "preprocessed"
    processed_files_dirname = "processed"

    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config: dict = yaml.safe_load(file)

        self.base_dirpath = self.get("base_dirpath")

    def get(self, key):
        return self.config[key]

    def get_train_raw_dataset_path(self):
        return self.get_dataset_path(
            self.raw_files_dirname, self.get("raw_train_filename"), self.get("raw_files_extension")
        )

    def get_test_raw_dataset_path(self):
        return self.get_dataset_path(
            self.raw_files_dirname, self.get("raw_test_filename"), self.get("raw_files_extension")
        )

    def get_preprocessed_dataset_path(self):
        return self.get_dataset_path(
            self.preprocessed_files_dirname,
            self.get("preprocessed_dirname"),
        )

    def get_processed_dataset_path(self):
        return self.get_dataset_path(
            self.processed_files_dirname,
            self.get("processed_dirname"),
        )

    def get_dataset_path(self, level, filename, file_extension=None):
        if file_extension is not None:
            filename = f"{filename}.{file_extension}"
        return os.path.join(self.base_dirpath, self.data_dirname, level, filename)
