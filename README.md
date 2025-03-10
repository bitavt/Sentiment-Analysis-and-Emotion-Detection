# Fine tuning distilBERT for sentiment analysis

## Overview
This project fine-tunes **DistilBERT**, a lightweight transformer model, on the Stanford Sentiment Treebank (movie reviews) data set for the sentiment analysis task using the Hugging Face `Trainer` API. The pipeline includes dataset preparation, model training, evaluation, and performance metrics computation.

## Workflow

### 1. **Define Sentiment Dataset Class**
The `SentimentDataset` class is a PyTorch `Dataset` wrapper that allows for efficient batch processing with tokenized text data and corresponding sentiment labels.

### 2. **Define HuggingFaceSentimentTrainer Class**
This class encapsulates:
- **Model and Tokenizer Initialization**
- **Data Preprocessing**
- **Training Pipeline using `Trainer` API**
- **Model Evaluation and Performance Metrics Computation**

### 3. **Tokenization Function**
The  function prepares text input for the DistilBERT model by tokenizing and padding them to the maximum sequence length.

### 4. **Prepare Training, Validation, and Test Data**
The dataset is split as follows:
- **60% for Training**
- **20% for Validation**
- **20% for Testing**

Each split is transformed into a list of strings and then tokenized separately with `tokenize_texts()`, and finally datasets are wrapped into `SentimentDataset` objects.

### 5. **Train the Model**
- The training process is handled by Hugging Face's `Trainer` class.
- Training hyperparameters such as learning rate, batch size, and number of epochs are configurable.
- The best model is saved based on evaluation performance.

### 6. **Evaluate the Model**
After training, `trainer.evaluate()` is used to assess performance on the validation set.

### 7. **Compute Performance Metrics**
Metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The predictions on the test set are compared against ground truth labels, and results are stored in a Pandas DataFrame.


