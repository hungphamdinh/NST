import json
import os
from datasets import DatasetDict, Dataset
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------
# 1. Load JSONL Data
# ------------------------------

def load_jsonl(file_path):
    """Load a JSONL file and extract text with main categories."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            
            # Extract main categories from all labels
            categories = []
            for label in item.get('labels', []):
                if len(label) == 3:  # Ensure proper label format [start, end, label]
                    category = label[2].split('#')[0]  # Get part before '#'
                    categories.append(category)
            
            # Use the first category if available, otherwise 'Others'
            label = categories[0] if categories else 'Others'
            data.append({'text': text, 'label': label})
    
    return data

# Load all splits from data/VN_FB directory
data_dir = "data/VN_FB"
train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
dev_data = load_jsonl(os.path.join(data_dir, "dev.jsonl"))
test_data = load_jsonl(os.path.join(data_dir, "test.jsonl"))

# Create label mappings (automatically based on all data)
all_labels = [item['label'] for item in train_data + dev_data + test_data]
unique_labels = sorted(list(set(all_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {v: k for k, v in label2id.items()}

# Convert to DatasetDict
datasets = DatasetDict({
    'train': Dataset.from_list(train_data),
    'validation': Dataset.from_list(dev_data),
    'test': Dataset.from_list(test_data)
})

# Map labels to IDs
def map_labels(example):
    return {'label': label2id[example['label']]}

datasets = datasets.map(map_labels)

# -------------------------------
# 2. Tokenization
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples['text']]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# -------------------------------
# 3. Model Training
# -------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,      # Larger batch for medium-sized datasets
    per_device_eval_batch_size=32,       # Faster evaluation
    num_train_epochs=5,                  # Fewer epochs (prevents overfitting)
    learning_rate=3e-5,                  # Slightly higher than default
    weight_decay=0.02,                   # Stronger regularization
    eval_strategy="steps",          # Evaluate every N steps
    eval_steps=200,                      # Evaluate 2-3 times per epoch
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    gradient_accumulation_steps=1,       # No need for accumulation
    warmup_ratio=0.1,                    # 10% warmup
    optim="adamw_torch",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# -------------------------------
# 4. Evaluation & Saving
# -------------------------------

# Evaluate on test set
test_results = trainer.evaluate(tokenized_datasets["test"])
print("\nTest Results:", test_results)

# Save model
save_dir = "./phobert_feedback_finetuned"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")

# Save label mappings
with open(os.path.join(save_dir, "label_mappings.json"), 'w') as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False)