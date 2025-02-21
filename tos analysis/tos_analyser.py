import os
import re
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Determine device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

### PART 1: Update CSV Content from Local Files with Regex Filtering

# Updated paths for your local system
merged_csv_path = 'merged_output.csv'
# Folder containing the text files (each file named like "<name>.txt")
local_files_folder = 'text'

# Load the merged CSV
df = pd.read_csv(merged_csv_path)

def read_and_filter_content(name):
    """
    Constructs the file path (expects <name>.txt), reads its content,
    and filters it to keep only English letters, digits, and whitespace.
    """
    file_path = os.path.join(local_files_folder, f"{name}.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Remove any character that is not an English letter, digit, or whitespace.
        filtered_content = re.sub(r'[^A-Za-z0-9\s]', '', content)
        return filtered_content
    else:
        return ''

# Update the 'content' column by reading and filtering the corresponding file
df['content'] = df['name'].apply(read_and_filter_content)

# For training, rename 'content' to 'tos'
df.rename(columns={'content': 'tos'}, inplace=True)

# Convert risk_score into a binary label.
df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
risk_threshold = 50  # adjust threshold as needed
df['label'] = df['risk_score'].apply(lambda x: 1 if x >= risk_threshold else 0)

# Optionally, save the updated CSV before training
updated_csv_path = 'merged_output_updated.csv'
df.to_csv(updated_csv_path, index=False)
print("Updated CSV with TOS text and labels saved to:", updated_csv_path)

### PART 2: Prepare Data & Train BERT Model for TOS Classification

# Split the data into training and testing sets.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Initialize the tokenizer and model
model_name = "bert-base-uncased"  # or a domain-specific model if available
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)  # Move model to GPU if available

# Define tokenization function for the TOS text
def tokenize_function(examples):
    return tokenizer(examples["tos"], truncation=True, padding="max_length", max_length=256)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the dataset format for PyTorch tensors (we need input_ids, attention_mask, and label)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments with updated output directories
training_args = TrainingArguments(
    output_dir=r'\results',  # directory to save model checkpoints
    num_train_epochs=2,                                    # adjust epochs as needed
    per_device_train_batch_size=8,                         # training batch size
    per_device_eval_batch_size=8,                          # evaluation batch size
    evaluation_strategy="epoch",                           # evaluation strategy
    logging_dir=r'\logs',    # directory for logs
    logging_steps=10,
    save_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
print("Training model...")
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate()
print("Evaluation Results:", results)

### PART 3: Save the Fine-Tuned Model and Tokenizer for FastAPI Usage

# Define the directory where the model will be saved
model_save_path = './fastapi_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("Model and tokenizer saved to:", model_save_path)

### PART 4: Use the Trained Model to Predict Fault on Each TOS Text

def predict_fraud_score(tos_text):
    """
    Given a TOS text, returns the fraud probability score using the fine-tuned model.
    """
    model.eval()  # Set model to evaluation mode
    inputs = tokenizer(tos_text.lower(), return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    # Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    fraud_probability = probabilities[0][1].item()  # label 1 is considered 'fault'
    return fraud_probability

# Create a new column "prediction_fault" based on model predictions.
df['prediction_fault'] = df['tos'].apply(lambda x: 1 if predict_fraud_score(x) > 0.5 else 0)

# Save the final DataFrame with the prediction_fault column to a new CSV file.
final_csv_path = '\merged_output_with_prediction_fault.csv'
df.to_csv(final_csv_path, index=False)
print("Final output with prediction_fault saved to:", final_csv_path)
