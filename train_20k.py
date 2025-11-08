"""
Training with 20K samples for better quality
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
import json


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


def tokenize_function(examples, tokenizer, max_length=256):
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


BASE_MODEL = "gpt2"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES = 20000  # 20K samples
OUTPUT_DIR = "./chatbot-20k"
MAX_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

print("="*80)
print(f"Training with {NUM_SAMPLES} samples (20K)")
print("="*80)

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

print(f"\n2. Loading {NUM_SAMPLES} samples...")
dataset = load_dataset(DATASET_NAME, split="train_sft", streaming=False)
dataset = dataset.select(range(NUM_SAMPLES))
print(f"   ✓ Loaded: {len(dataset)} examples")

print("\n3. Tokenizing...")
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)
print("   ✓ Done")

print("\n4. Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=200,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print(f"\n5. Training for {NUM_EPOCHS} epochs...")
print(f"   Total steps: ~{len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")
print("="*80)

trainer.train()

print("\n6. Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metadata = {
    "base_model": BASE_MODEL,
    "dataset": DATASET_NAME,
    "num_samples": NUM_SAMPLES,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "model_parameters": model.num_parameters(),
}

with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")
