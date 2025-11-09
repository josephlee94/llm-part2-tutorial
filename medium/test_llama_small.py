"""
Test script - Exact copy of train_chatbot.py approach but with TinyLlama
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Simple formatting for Alpaca
def format_alpaca(example):
    if example["input"]:
        text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        text = f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""
    return text

def tokenize_function(examples, tokenizer, max_length=256):
    # Format each example
    texts = [format_alpaca({"instruction": examples["instruction"][i],
                            "input": examples["input"][i],
                            "output": examples["output"][i]})
             for i in range(len(examples["instruction"]))]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
NUM_SAMPLES = 1000  # Small for testing
MAX_LENGTH = 256

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print("Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

print("Tokenizing...")
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)

print("Setting up training...")
training_args = TrainingArguments(
    output_dir="./test-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=50,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    bf16=torch.cuda.is_available(),
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
    max_steps=100,  # Just 100 steps for testing
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Done!")
