import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
# custom imports
from constants import *



class SentimentDataset(torch.utils.data.Dataset):
    """
    Extends torch.utils.data.Dataset, making it compatible with PyTorch’s DataLoader for efficient batch processing.

    Args:
        encodings: A dictionary containing tokenized text data
        labels: A list or array of sentiment labels corresponding to the encoded text.

    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Retrieve the tokenized input for a given index and store it in a dictionary
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Convert the label to a PyTorch tensor and add it to the dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)


class HuggingFaceSentimentTrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        """
        Initializes the trainer by loading the model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face.
            num_labels (int): The number of labels for classification (e.g., 2 for binary classification).
        """
        self.model_name = model_name
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # load the model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_texts(self, texts):
        """
        Tokenizes the input text.

        Args:
            texts (list of str): List of text inputs.

        Returns:
            Tokenized input text.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        return encodings

    def preprare_input_for_trainer(self, df, test_size=.4):

        # 60% train, 20% validation and 20% test
        train_df, temp_df = train_test_split(
            df,
            test_size= test_size,
            random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )
        print(f"Training Set: {len(train_df):,} samples")
        print(f"Validation Set: {len(val_df):,} samples")
        print(f"Test Set: {len(test_df):,} samples")

        # convert the column to a list of strings and pass it to tokenizer
        # train
        train_texts  = train_df["text"].tolist()
        train_encodings = self.tokenize_texts(train_texts)
        train_labels = train_df["label"].tolist()
        # validation
        val_texts = val_df["text"].tolist()
        val_encodings = self.tokenize_texts(val_texts)
        val_labels =  val_df["label"].tolist()
        # test
        test_texts =  test_df["text"].tolist()
        test_encodings = self.tokenize_texts(test_texts)
        test_labels= test_df["label"].tolist()

        # prepare data for the Hugging Face Trainer
        self.train_dataset = SentimentDataset(train_encodings, train_labels)
        self.val_dataset = SentimentDataset(val_encodings, val_labels)
        self.test_dataset = SentimentDataset(test_encodings, test_labels)

    def train_model(self,
              output_dir= OUTPUT_DIR,
              learning_rate=LEARNING_RATE,
              per_device_train_batch_size= PER_DEVICE_TRAIN_BATCH_SIZE,
              per_device_eval_batch_size= PER_DEVICE_EVAL_BATCH_SIZE,
              epochs= NUM_TRAIN_EPOCHS,
              seed= SEED
              ):
        """
        Trains the model using Hugging Face Trainer.

        Args:

        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate= learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=epochs,
            seed= seed
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        # train the model
        self.trainer.train()

        # evaluate the model's performance on a validation dataset
        results = self.trainer.evaluate()
        print(f"model's performance on a validation dataset: {results}")

    def compute_metrics(self):
        """
        Computes evaluation metrics.

        Returns:
            A dataframe containing accuracy, precision, recall, and F1 score.
        """
        # Get predictions for test set
        predictions = self.trainer.predict(self.test_dataset)
        # predicted label
        preds = predictions.predictions.argmax(-1)
        # true labels
        labels = predictions.label_ids

        # Calculate metrics
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=labels,
            y_pred=preds,
            average='binary'
        )

        evaluation_metrics = [
            ["Accuracy", accuracy],
            ["Precision", precision],
            ["Recall", recall],
            ["F1 Score", f1],
        ]
        metric_df= pd.DataFrame(evaluation_metrics, columns=["metric", "value"])
        return metric_df

