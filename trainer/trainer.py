from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

# Change working directory (if needed)
os.chdir(r"C:\Users\Josh\Desktop\programming\AI\DeepSeek\chiccenAI\trainer")

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load your chat logs from the text file
with open("./trainingdata3.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Preprocess the data: remove empty lines and strip whitespace
lines = [line.strip() for line in lines if line.strip()]

# Convert the text data into a Hugging Face Dataset
dataset = Dataset.from_dict({"text": lines})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Add labels for causal language modeling
tokenized_dataset = tokenized_dataset.map(
    lambda examples: {"labels": examples["input_ids"]},
    batched=True,
)

# Use a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False for causal language modeling
)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="../results",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Reduce epochs
    per_device_train_batch_size=16,  # Increase batch size
    gradient_accumulation_steps=2,  # Use gradient accumulation
    fp16=True,  # Enable mixed precision (requires CUDA)
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=500,
    eval_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("../fine-tuned-model")
tokenizer.save_pretrained("../fine-tuned-model")