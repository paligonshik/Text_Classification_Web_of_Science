import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_loader.data_loader import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
class TextClassificationPipeline:
    def __init__(self, data_source, dataset_id, tokenizer_model='bert-base-uncased', num_labels=11):
        self.data_loader = DataLoader(data_source, dataset_id)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.num_labels = num_labels
        self.train_dataset = None
        self.eval_dataset = None
        self.model = None
        self.trainer = None

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

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train_model(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
        self.get_trainer()
        self.trainer.train()

    def evaluate_model(self):
        results = self.trainer.evaluate()
        print("Evaluation Results:", results)

    def save_model(self, path='./saved_model'):
        self.model.save_pretrained(path)

    def load_model(self, path='./saved_model'):

        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self.get_trainer()

    def get_trainer(self):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_total_limit=1
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics
        )

    def preprocess_new_data(self, texts):
        """ This method preprocesses a list of text data into a format suitable for the model. """
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512,
                                add_special_tokens=True)
        return inputs

    def predict(self, texts):
        """ This method predicts the labels for a list of texts using the trained model. """
        processed_inputs = self.preprocess_new_data(texts)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**processed_inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label_indices = probabilities.argmax(dim=1).tolist()
        return predicted_label_indices, probabilities.tolist()


if  __name__ == "__main__":
    pipeline = TextClassificationPipeline(data_source='web_of_science', dataset_id='WOS5736')
    pipeline.load_and_prepare_data()
    pipeline.apply_tokenization()
    pipeline.load_model()
    pipeline.evaluate_model()