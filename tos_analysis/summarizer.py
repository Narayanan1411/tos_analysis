import torch
from transformers import BertTokenizer, EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize the EncoderDecoderModel using BERT as both encoder and decoder
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Set the special tokens
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Ensure the vocab size is set correctly
model.config.vocab_size = model.config.encoder.vocab_size

# Set some generation parameters (adjust as needed)
model.config.max_length = 128
model.config.min_length = 30
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Load a summarization dataset (using CNN/DailyMail as an example)
# For demonstration, we're using a small subset (1%) of the training split.
dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")

def tokenize_function(example):
    # Tokenize the input (article) and target (highlights) texts.
    inputs = tokenizer(example["article"], truncation=True, max_length=512)
    outputs = tokenizer(example["highlights"], truncation=True, max_length=128)
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Tokenize the dataset; remove original columns to save space.
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["article", "highlights", "id"])

# Define training arguments. Adjust hyperparameters as needed.
training_args = Seq2SeqTrainingArguments(
    output_dir="./bert2bert-summarizer",
    num_train_epochs=1,  # Increase the number of epochs for real training
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if a GPU is available
)

# Initialize the Trainer for sequence-to-sequence tasks.
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Begin fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./bert2bert-summarizer")
tokenizer.save_pretrained("./bert2bert-summarizer")
