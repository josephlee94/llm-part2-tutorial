"""
Quick test script to verify the pipeline works with minimal data.
This is a lightweight version for testing purposes.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def format_chat_prompt(example):
    messages = example.get("messages", [])
    formatted_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            formatted_text += f"User: {content}\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n"
    return formatted_text


def tokenize_function(examples, tokenizer, max_length=128):
    texts = []
    for i in range(len(examples["messages"])):
        example = {"messages": examples["messages"][i]}
        formatted = format_chat_prompt(example)
        texts.append(formatted)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Configuration for quick test
BASE_MODEL = "gpt2"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES = 100  # Very small for quick test
OUTPUT_DIR = "./chatbot-test"
MAX_LENGTH = 128

print("🧪 Quick Test: Verifying pipeline...")
print(f"   Using {NUM_SAMPLES} samples")
print(f"   Model: {BASE_MODEL}")

# Load model
print("\n1. Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print(f"   ✓ Model loaded: {model.num_parameters():,} parameters")

# Load dataset
print("\n2. Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train_sft", streaming=False)
dataset = dataset.select(range(NUM_SAMPLES))
print(f"   ✓ Dataset loaded: {len(dataset)} examples")

# Tokenize
print("\n3. Tokenizing...")
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)
print("   ✓ Tokenization complete")

# Training args
print("\n4. Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    bf16=torch.cuda.is_available(),  # Use bf16 instead of fp16 to avoid gradient issues
    report_to=[],
    remove_unused_columns=False,
    max_steps=20,  # Only 20 steps for quick test
    gradient_checkpointing=False,
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

print("   ✓ Trainer ready")

# Train
print("\n5. Running quick training (20 steps)...")
trainer.train()

print("\n6. Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Quick test complete!")
print(f"   Model saved to: {OUTPUT_DIR}")
print("\n🎯 Pipeline verified! Ready for full training.")
