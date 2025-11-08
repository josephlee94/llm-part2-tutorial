"""
Supervised Fine-Tuning (SFT) Script: Transform a Vanilla LLM into a Chatbot

This script demonstrates how to take a base/vanilla language model and convert it
into an instruction-following chatbot using supervised fine-tuning on a chat dataset.
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
from typing import Dict, List
import json


def format_chat_prompt(example: Dict) -> str:
    """
    Format a conversation into a training prompt.

    This function converts chat messages into a format suitable for training.
    We use a simple template: User: ... Assistant: ...
    """
    messages = example.get("messages", [])

    if not messages:
        return ""

    formatted_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            formatted_text += f"User: {content}\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n"

    return formatted_text


def prepare_dataset(dataset_name: str = "HuggingFaceH4/ultrachat_200k",
                    num_samples: int = 5000,
                    split: str = "train_sft"):
    """
    Download and prepare a chat dataset for supervised fine-tuning.

    Args:
        dataset_name: HuggingFace dataset identifier
        num_samples: Number of samples to use (smaller for faster training)
        split: Which split to use from the dataset

    Returns:
        Processed dataset ready for training
    """
    print(f"📥 Downloading dataset: {dataset_name}")
    print(f"   Using {num_samples} samples for training...")

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split, streaming=False)

    # Take a subset for faster training
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"✅ Dataset loaded: {len(dataset)} examples")

    return dataset


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    """
    Tokenize the formatted chat examples.

    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    # Format each example into a chat prompt
    texts = []
    for i in range(len(examples["messages"])):
        example = {"messages": examples["messages"][i]}
        formatted = format_chat_prompt(example)
        texts.append(formatted)

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


def main():
    """
    Main training pipeline:
    1. Load base/vanilla model
    2. Download chat dataset
    3. Prepare dataset for training
    4. Fine-tune the model
    5. Save the fine-tuned chatbot
    """

    # ============================================
    # Configuration
    # ============================================
    BASE_MODEL = "gpt2"  # Vanilla/base model (you can also use "EleutherAI/pythia-160m" or others)
    DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
    NUM_TRAIN_SAMPLES = 5000  # Use subset for faster training
    OUTPUT_DIR = "./chatbot-finetuned"
    MAX_LENGTH = 256  # Maximum sequence length (shorter for faster training)

    # Training hyperparameters
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    SAVE_STEPS = 500

    print("=" * 60)
    print("🚀 Supervised Fine-Tuning: Vanilla LLM → Chatbot")
    print("=" * 60)

    # ============================================
    # Step 1: Load Base/Vanilla Model
    # ============================================
    print(f"\n📦 Step 1: Loading base model: {BASE_MODEL}")
    print("   This is the 'vanilla' pretrained model (like GPT before ChatGPT)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    print(f"   Device: {next(model.parameters()).device}")

    # ============================================
    # Step 2: Download and Prepare Dataset
    # ============================================
    print(f"\n📥 Step 2: Downloading chat dataset for SFT")
    dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        num_samples=NUM_TRAIN_SAMPLES
    )

    # Show a sample
    print("\n📝 Sample conversation:")
    print("-" * 60)
    sample = dataset[0]
    print(format_chat_prompt(sample))
    print("-" * 60)

    # ============================================
    # Step 3: Tokenize Dataset
    # ============================================
    print(f"\n🔤 Step 3: Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    print(f"✅ Tokenization complete")

    # ============================================
    # Step 4: Set Up Training
    # ============================================
    print(f"\n⚙️  Step 4: Setting up training configuration...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=50,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # Use bfloat16 for better stability on modern GPUs
        report_to=[],  # Disable wandb/tensorboard for simplicity
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # ============================================
    # Step 5: Train the Model
    # ============================================
    print(f"\n🏋️  Step 5: Starting supervised fine-tuning...")
    print(f"   Training for {NUM_EPOCHS} epochs")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Total training steps: ~{len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training!
    print("\n" + "=" * 60)
    print("🎯 TRAINING STARTED")
    print("=" * 60 + "\n")

    trainer.train()

    # ============================================
    # Step 6: Save the Fine-Tuned Model
    # ============================================
    print("\n💾 Step 6: Saving fine-tuned chatbot model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Model saved to: {OUTPUT_DIR}")

    # Save training metadata
    metadata = {
        "base_model": BASE_MODEL,
        "dataset": DATASET_NAME,
        "num_samples": NUM_TRAIN_SAMPLES,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "model_parameters": model.num_parameters(),
    }

    with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✨ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Fine-tuned chatbot saved to: {OUTPUT_DIR}")
    print(f"📊 Training metadata saved to: {OUTPUT_DIR}/training_metadata.json")
    print(f"\n🎉 You've successfully transformed a vanilla LLM into a chatbot!")
    print(f"   Run 'python test_chatbot.py' to compare base vs fine-tuned models")


if __name__ == "__main__":
    main()
