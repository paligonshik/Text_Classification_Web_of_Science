from data_loader.data_loader import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

class TextClassificationPipeline:
    def __init__(self, data_source, dataset_id, tokenizer_model='bert-base-uncased', num_labels=11):
        self.data_loader = DataLoader(data_source, dataset_id)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.num_labels = num_labels
        self.train_dataset = None
        self.eval_dataset = None

    def load_and_prepare_data(self):
        self.data_loader.load_data()
        self.data_loader.split_train_test()
        train_data = self.data_loader.get_train_data()
        eval_data = self.data_loader.get_test_data()

        # Modify datasets
        self.train_dataset = train_data.map(
            lambda example: {'input_data': example['input_data'], 'label': example['label']},
            remove_columns=['label_level_1', 'label_level_2']
        )

        self.eval_dataset = eval_data.map(
            lambda example: {'input_data': example['input_data'], 'label': example['label']},
            remove_columns=['label_level_1', 'label_level_2']
        )

    def preprocess_function(self, examples):
        return self.tokenizer(examples['input_data'], truncation=True, padding='max_length', max_length=512)

    def apply_tokenization(self):
        self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        self.eval_dataset = self.eval_dataset.map(self.preprocess_function, batched=True)

    def train_model(self):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_total_limit=1
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        trainer.train()


if __name__ == "__main__":
    pipeline = TextClassificationPipeline("web_of_science", "WOS5736")
    pipeline.load_and_prepare_data()
    pipeline.apply_tokenization()
    pipeline.train_model()
