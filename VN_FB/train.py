import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------
# 1. Load Only the Needed Columns from Excel
# ------------------------------

# Update this file path to point to your actual Excel file.
excel_file = 'feedback.xlsx'

# Load the Excel file, but only the columns you need.
df = pd.read_excel(excel_file, usecols=['Type', 'Description'])

# Preview the first few rows to confirm structure
print("Data Sample:")
print(df.head())

# Rename columns to match our standard "label" and "text" usage.
df.rename(columns={'Type': 'label', 'Description': 'text'}, inplace=True)

# The 'label' column may have values such as:
#   - "Others"
#   - "Suggestion"
#   - "Complaint update name"
#   - etc.
#
# We need to map these to integer IDs. Let's see unique values:
unique_labels = df['label'].unique()
print("Unique labels found:", unique_labels)

# Suppose we have three broad categories you want to model:
#   1) "Others"
#   2) "Suggestion"
#   3) "Complaint update name"
# You can define a mapping. If there are more categories, include them here.
# Adjust the mapping to match exactly the set of unique labels you want to handle.
label2id = {
    "Others": 0,
    "Suggestion": 1,
    "Complaint update name": 2
    # Add more mappings if you have additional label types
}
id2label = {v: k for k, v in label2id.items()}

# Map the label column to integer IDs. If any label does not exist in label2id,
# it will become NaN and needs to be handled or added to the mapping above.
df['label'] = df['label'].map(label2id)

# Drop rows with NaN labels (if any labels didn't match the mapping).
df.dropna(subset=['label'], inplace=True)

# -------------------------------
# 2. Split the Data
# -------------------------------

# Split into train and validation sets (e.g., 80% train, 20% validation).
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Convert DataFrames to Hugging Face Datasets.
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# -------------------------------
# 3. Tokenize Using PhoBERT
# -------------------------------

# Load PhoBERT tokenizer (base). We disable fast mode for better compatibility with Vietnamese.
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def tokenize_function(example):
    # Convert all elements in 'text' to strings (handles numbers, None, etc.)
    texts = [str(text) if text is not None else "" for text in example['text']]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Apply tokenization over the entire dataset.
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# (Optional) Remove the original 'text' column to save memory.
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Set the dataset format to PyTorch.
tokenized_datasets.set_format("torch")

# -------------------------------
# 4. Fine-Tune PhoBERT Model
# -------------------------------

# Load the PhoBERT model for sequence classification.
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./phobert_feedback_results",
    per_device_train_batch_size=8,  # Start with 8 or 16
    per_device_eval_batch_size=8,
    num_train_epochs=10,           # More epochs for small data
    learning_rate=2e-5,            # PhoBERT typically uses 1e-5 to 5e-5
    weight_decay=0.01,             # Regularization
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

# Define our compute_metrics function for evaluation.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# Evaluate the model after training.
results = trainer.evaluate()
print("\nValidation Results:", results)

# -------------------------------
# 5. Save the Fine-Tuned Model
# -------------------------------

save_directory = "./phobert_feedback_finetuned"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved to {save_directory}")