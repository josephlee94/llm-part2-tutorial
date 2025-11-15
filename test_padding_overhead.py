"""
Test to see how much padding overhead exists with different batch sizes
"""
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
import torch
from datasets import load_dataset
import numpy as np

model_name = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load small dataset
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
dataset = load_dataset(DATASET_NAME, split="train_sft[:100]", streaming=False)

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
    formatted_text += tokenizer.eos_token
    return formatted_text

def tokenize_function(example):
    text = format_chat_prompt(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"][:]
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"],
    }

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    batched=False
)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8
)

print("="*80)
print("PADDING OVERHEAD ANALYSIS")
print("="*80)

# Test different batch sizes
batch_sizes = [1, 2, 4]

for batch_size in batch_sizes:
    print(f"\n{'='*80}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*80}")

    # Sample random batches
    padding_ratios = []
    original_lengths = []
    padded_lengths = []

    for i in range(0, min(50, len(tokenized_dataset)), batch_size):
        batch_items = [tokenized_dataset[j] for j in range(i, min(i + batch_size, len(tokenized_dataset)))]

        # Get original lengths
        orig_lengths = [len(item['input_ids']) for item in batch_items]
        original_lengths.extend(orig_lengths)

        # Create batch with collator
        batch = collator(batch_items)

        # Get padded length
        padded_length = batch['input_ids'].shape[1]
        padded_lengths.extend([padded_length] * len(batch_items))

        # Calculate padding
        total_original_tokens = sum(orig_lengths)
        total_padded_tokens = padded_length * len(batch_items)
        padding_ratio = (total_padded_tokens - total_original_tokens) / total_padded_tokens
        padding_ratios.append(padding_ratio)

    avg_padding_ratio = np.mean(padding_ratios)
    avg_original_length = np.mean(original_lengths)
    avg_padded_length = np.mean(padded_lengths)

    print(f"Average original sequence length: {avg_original_length:.1f} tokens")
    print(f"Average padded length: {avg_padded_length:.1f} tokens")
    print(f"Average padding overhead: {avg_padding_ratio*100:.1f}% of total tokens are padding")
    print(f"Wasted computation: {avg_padding_ratio*100:.1f}% of forward/backward pass")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nPadding overhead increases with batch size because:")
print("- Batch 1: Only pad to multiple of 8")
print("- Batch 2+: Pad all sequences to match the longest in batch")
print("\nMore padding = More wasted computation = Slower training!")
print("="*80)
