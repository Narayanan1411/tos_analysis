import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# -----------------------------
# Step 1: Load the Extensive Dataset
# -----------------------------
dataset_file = "extensive_tos_access_rules.json"

# Ensure the file exists
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"{dataset_file} not found. Please save the JSON dataset in this file.")

with open(dataset_file, "r") as f:
    dataset_json = json.load(f)

records = dataset_json["TOS_Access_Rules"]

# Convert records to a Pandas DataFrame
df = pd.DataFrame(records)

# Create a single text field from key components using [SEP] tokens
df["text"] = (
    df["data_category"] + " [SEP] " +
    df["department"] + " [SEP] " +
    df["tos_clause"] + " [SEP] " +
    df["legal_basis"]
)

# Convert boolean label to integer (True -> 1, False -> 0)
df["label"] = df["access_allowed"].astype(int)

# Split into training and testing sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# -----------------------------
# Step 2: Load Model and Tokenizer
# -----------------------------
model_checkpoint = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# -----------------------------
# Step 3: Setup GPU and Move Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# -----------------------------
# Step 4: Tokenize the Dataset
# -----------------------------
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove unused columns
columns_to_remove = [
    "id", "data_category", "department", "tos_clause", "legal_basis", 
    "access_allowed", "text", "__index_level_0__"
]
tokenized_datasets = tokenized_datasets.remove_columns(
    [col for col in columns_to_remove if col in tokenized_datasets["train"].column_names]
)
tokenized_datasets.set_format("torch")

# -----------------------------
# Step 5: Setup TrainingArguments and Trainer for GPU (with fp16)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    fp16=True if device.type == "cuda" else False  # Enable mixed precision if using GPU
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -----------------------------
# Step 6: Fine-Tune the Model on GPU
# -----------------------------
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save the fine-tuned model and tokenizer
tokenizer.save_pretrained("./results")
model.save_pretrained("./results")

# -----------------------------
# Step 7: Inference Testing on Sample Inputs
# -----------------------------
sample_texts = [
    # Sample 1: Medical department should have access to Health Records.
    "Health_Records [SEP] Medical [SEP] Medical staff are granted access to health records to provide healthcare services in compliance with HIPAA regulations, as specified in the Terms of Service. [SEP] Healthcare provision and HIPAA compliance",
    # Sample 2: Finance department should NOT have access to Health Records.
    "Health_Records [SEP] Finance [SEP] Access to health records is strictly limited to authorized Medical and Legal & Compliance departments, and is not permitted for Finance. [SEP] Regulatory compliance",
    # Sample 3: Customer Support should have access to Personal Information.
    "Personal_Information [SEP] Customer Support [SEP] Customer Support personnel are permitted to access personal information for account management and support purposes as stated in the Terms of Service. [SEP] Contract performance and user consent"
]

print("\n--- Inference on Sample Inputs ---")
for idx, text in enumerate(sample_texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    result = "Access Allowed" if prediction == 1 else "Access Denied"
    print(f"Sample {idx+1} Prediction: {result}")