from datasets import load_dataset
from loguru import logger


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


# Usage of DataLoader
if __name__ == "__main__":
    data_loader = DataLoader
    data_loader.load_data()
    data_loader.split_train_test()  # Can use a ratio for flexibility
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()
