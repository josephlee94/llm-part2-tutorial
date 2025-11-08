"""
Demo Training Script: Quick demonstration of the full SFT pipeline
Uses 1000 samples for a balance between speed and quality demonstration.
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


# Configuration
BASE_MODEL = "gpt2"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES = 1000  # Demo with 1000 samples
OUTPUT_DIR = "./chatbot-demo"
MAX_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
WARMUP_STEPS = 50

print("=" * 70)
print("🎬 DEMO: Supervised Fine-Tuning (1000 samples, 2 epochs)")
print("=" * 70)

# Load model
print(f"\n1️⃣  Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print(f"   ✅ Loaded: {model.num_parameters():,} parameters")

# Load dataset
print(f"\n2️⃣  Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train_sft", streaming=False)
dataset = dataset.select(range(NUM_SAMPLES))
print(f"   ✅ Loaded: {len(dataset)} examples")

# Show sample
print("\n   📝 Sample conversation:")
print("   " + "-" * 60)
sample_text = format_chat_prompt(dataset[0])
for line in sample_text.split('\n')[:4]:  # Show first few lines
    print(f"   {line}")
print("   " + "-" * 60)

# Tokenize
print(f"\n3️⃣  Tokenizing dataset...")
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)
print(f"   ✅ Tokenization complete")

# Training setup
print(f"\n4️⃣  Configuring training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
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

steps = len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
print(f"   ✅ Training configuration:")
print(f"      - Epochs: {NUM_EPOCHS}")
print(f"      - Batch size: {BATCH_SIZE}")
print(f"      - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"      - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"      - Estimated steps: ~{steps}")
print(f"      - Learning rate: {LEARNING_RATE}")

# Train
print(f"\n5️⃣  Training started...")
print("=" * 70)
trainer.train()
print("=" * 70)

# Save
print(f"\n6️⃣  Saving model to {OUTPUT_DIR}...")
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

print(f"   ✅ Model saved!")

print("\n" + "=" * 70)
print("✨ DEMO TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📁 Fine-tuned model: {OUTPUT_DIR}")
print(f"📊 Metadata: {OUTPUT_DIR}/training_metadata.json")
print(f"\n🎯 Next steps:")
print(f"   1. Test the model:")
print(f"      python test_chatbot.py --mode compare --finetuned-model {OUTPUT_DIR}")
print(f"   2. Chat interactively:")
print(f"      python test_chatbot.py --mode interactive --finetuned-model {OUTPUT_DIR}")
print(f"   3. Run full training with more data:")
print(f"      python train_chatbot.py")
