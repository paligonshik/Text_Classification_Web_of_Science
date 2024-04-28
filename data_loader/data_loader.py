import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from loguru import logger
import os
from torchtext.data import get_tokenizer
import torchtext.vocab as vocab

glove = vocab.GloVe(name='6B', dim=100)


class DataLoader:
    def __init__(self, dataset_name, subset=None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.original_dataset = None
        self.training_dataset = None
        self.testing_dataset = None

    def load_data(self):
        try:
            self.original_dataset = load_dataset(self.dataset_name, self.subset)['train']
            logger.info("Dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def split_train_test(self, test_size=1188):
        num_rows = len(self.original_dataset)
        if test_size < 1:  # Assuming test_size is a ratio if less than 1
            test_ratio = test_size
        else:
            test_ratio = test_size / num_rows

        try:
            train_test_split = self.original_dataset.train_test_split(test_size=test_ratio)
            self.training_dataset = train_test_split['train']
            self.testing_dataset = train_test_split['test']
            logger.info("Dataset split into train and test sets successfully.")
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            raise

    def get_train_data(self):
        return self.training_dataset

    def get_test_data(self):
        return self.testing_dataset

    def tokenize_and_vectorize(self, batch):
        tokenizer = get_tokenizer("basic_english")
        tokenized_texts = [tokenizer(text) for text in batch['input_data']]
        vectorized_texts = [[glove[token] for token in text if token in glove.stoi] for text in tokenized_texts]
        return {"vectorized_data": vectorized_texts}

    def apply_tokenization(self):
        self.training_dataset = self.training_dataset.map(self.tokenize_and_vectorize, batched=True)
        self.testing_dataset = self.testing_dataset.map(self.tokenize_and_vectorize, batched=True)
        logger.info("Tokenization and vectorization applied successfully.")

    def prepare_data(self):
        self.load_data()
        self.split_train_test()
        self.apply_tokenization()
        return self.get_train_data(), self.get_test_data()

    @staticmethod
    def filter_by_labels(dataset, level_1, level_2):
        return dataset.filter(
            lambda example: example['label_level_1'] == level_1 and example['label_level_2'] == level_2)

    def get_unique_labels(self):
        unique_groups = set((row['label_level_1'], row['label_level_2']) for row in self.training_dataset)
        return unique_groups

    def get_grouped_datasets(self, dataset):
        unique_groups = self.get_unique_labels()
        grouped_datasets = {}
        for group_keys in unique_groups:
            grouped_datasets[group_keys] = self.filter_by_labels(dataset, *group_keys)
        return grouped_datasets

    def get_train_test_grouped_datasets(self):
        train_grouped_datasets = self.get_grouped_datasets(self.training_dataset)
        test_grouped_datasets = self.get_grouped_datasets(self.testing_dataset)
        return train_grouped_datasets, test_grouped_datasets

    def save_grouped_datasets_to_disk(self,path='data'):
        train_grouped_datasets, test_grouped_datasets = self.get_train_test_grouped_datasets()
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')
        for group_keys, train_grouped_datasets in train_grouped_datasets.items():
            group_dir = os.path.join(train_path, f"group_{group_keys[0]}_{group_keys[1]}")
            os.makedirs(group_dir, exist_ok=True)
            train_grouped_datasets.save_to_disk(group_dir)
            logger.info(f"Saved training dataset for group {group_keys} to {group_dir}")
        for group_keys, test_grouped_datasets in test_grouped_datasets.items():
            group_dir = os.path.join(test_path, f"group_{group_keys[0]}_{group_keys[1]}")
            os.makedirs(group_dir, exist_ok=True)
            test_grouped_datasets.save_to_disk(group_dir)
            logger.info(f"Saved testing dataset for group {group_keys} to {group_dir}")

    def load__grouped_datasets_from_disk(self, dataset_type='train', path='data',group_to_load=(0,1)):
        try:
            dataset = load_from_disk(os.path.join(path, dataset_type, f"group_{group_to_load[0]}_{group_to_load[1]}"))
            logger.info(f"Loaded {dataset_type} dataset for group {group_to_load} successfully.")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {dataset_type} dataset for group {group_to_load}: {e}")
            raise

    def save_to_disk(self, path='data'):
        self.training_dataset.save_to_disk(os.path.join(path, 'full_train'))
        self.testing_dataset.save_to_disk(os.path.join(path, 'full_test'))
        logger.info("Saved datasets to disk successfully.")

if __name__ == '__main__':
    data_loader = DataLoader('web_of_science', 'WOS5736')
    train_data, test_data = data_loader.prepare_data()
    data_loader.save_to_disk()
    data_loader.save_grouped_datasets_to_disk()

    train_grouped_datasets = data_loader.load__grouped_datasets_from_disk(dataset_type='train', group_to_load=(2,))
    test_grouped_datasets = data_loader.load__grouped_datasets_from_disk(dataset_type='test')
    full_train = load_from_disk('data/full_train')
    full_test = load_from_disk('data/full_test')