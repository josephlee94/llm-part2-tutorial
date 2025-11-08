"""
Complete Training Pipeline: Full Dataset + Side-by-Side Testing

This script:
1. Trains on the FULL UltraChat dataset (207K+ examples)
2. Saves the trained model
3. Tests with side-by-side comparison of vanilla vs fine-tuned
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
    """
    Convert messages array to training format:
    Input: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Output: "User: ...\nAssistant: ...\n"
    """
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
    """Tokenize the formatted conversations"""
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

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized


def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate a response from a model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_models(base_model_name, finetuned_model_path):
    """Side-by-side comparison of vanilla vs fine-tuned"""
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON: Vanilla GPT-2 vs Fine-Tuned Chatbot")
    print("="*80)

    # Test prompts
    test_prompts = [
        "User: What is the capital of France?\nAssistant:",
        "User: Can you explain what machine learning is?\nAssistant:",
        "User: Write a Python function to calculate fibonacci numbers.\nAssistant:",
        "User: What are some tips for staying healthy?\nAssistant:",
    ]

    # Load vanilla model
    print("\nLoading vanilla GPT-2...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()

    # Load fine-tuned model
    print(f"Loading fine-tuned model from {finetuned_model_path}...")
    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    ft_tokenizer.pad_token = ft_tokenizer.eos_token
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ft_model.eval()

    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print("\n" + "="*80)
        print(f"TEST {i}/{len(test_prompts)}")
        print("="*80)
        print(f"\nPrompt: {prompt}\n")

        # Vanilla model
        print("BEFORE (Vanilla GPT-2):")
        print("-"*80)
        base_output = generate_response(base_model, base_tokenizer, prompt)
        print(base_output)
        print()

        # Fine-tuned model
        print("AFTER (Fine-Tuned with Full Dataset):")
        print("-"*80)
        ft_output = generate_response(ft_model, ft_tokenizer, prompt)
        print(ft_output)
        print()

    print("="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


def main():
    # Configuration
    BASE_MODEL = "gpt2"
    DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
    OUTPUT_DIR = "./chatbot-full-dataset"
    MAX_LENGTH = 256
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500

    print("="*80)
    print("FULL DATASET TRAINING: Vanilla LLM → Chatbot")
    print("="*80)

    # ============================================
    # STEP 1: Load Base Model
    # ============================================
    print(f"\n📦 STEP 1: Loading base model '{BASE_MODEL}'")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"   ✓ Model loaded: {model.num_parameters():,} parameters")
    print(f"   ✓ Device: {next(model.parameters()).device}")

    # ============================================
    # STEP 2: Load FULL Dataset
    # ============================================
    print(f"\n📥 STEP 2: Loading FULL dataset '{DATASET_NAME}'")
    dataset = load_dataset(DATASET_NAME, split="train_sft", streaming=False)

    print(f"   ✓ Total examples: {len(dataset):,}")

    # Show example of data preprocessing
    print("\n📝 Example of data preprocessing:")
    print("-"*80)
    print("RAW DATA:")
    print(f"  messages: {dataset[0]['messages'][:2]}")  # Show first 2 messages
    print("\nPREPROCESSED FORMAT:")
    print(format_chat_prompt(dataset[0])[:500] + "...")
    print("-"*80)

    # ============================================
    # STEP 3: Tokenize Dataset
    # ============================================
    print(f"\n🔤 STEP 3: Tokenizing {len(dataset):,} examples...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    print(f"   ✓ Tokenization complete")

    # ============================================
    # STEP 4: Configure Training
    # ============================================
    print(f"\n⚙️  STEP 4: Configuring training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=200,
        save_steps=5000,
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

    total_steps = len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    print(f"   ✓ Training configuration:")
    print(f"      - Dataset size: {len(dataset):,} examples")
    print(f"      - Epochs: {NUM_EPOCHS}")
    print(f"      - Batch size: {BATCH_SIZE}")
    print(f"      - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"      - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"      - Total steps: ~{total_steps:,}")
    print(f"      - Learning rate: {LEARNING_RATE}")

    # ============================================
    # STEP 5: Train
    # ============================================
    print(f"\n🏋️  STEP 5: Starting training on FULL dataset...")
    print("="*80)

    trainer.train()

    print("\n" + "="*80)

    # ============================================
    # STEP 6: Save Model
    # ============================================
    print(f"\n💾 STEP 6: Saving model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metadata = {
        "base_model": BASE_MODEL,
        "dataset": DATASET_NAME,
        "num_samples": len(dataset),
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "model_parameters": model.num_parameters(),
    }

    with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Model saved")
    print(f"   ✓ Metadata saved to {OUTPUT_DIR}/training_metadata.json")

    # ============================================
    # STEP 7: Test and Compare
    # ============================================
    print(f"\n🧪 STEP 7: Testing models...")
    test_models(BASE_MODEL, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✨ COMPLETE PIPELINE FINISHED!")
    print("="*80)
    print(f"\n📁 Fine-tuned model: {OUTPUT_DIR}")
    print(f"📊 Training samples: {len(dataset):,}")
    print(f"🎯 Ready to use!")


if __name__ == "__main__":
    main()
