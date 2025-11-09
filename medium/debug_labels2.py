"""
Debug script #2 - test DataCollatorForLanguageModeling WITHOUT pre-made labels
"""
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\nLoading small dataset sample...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:2]")

def format_instruction(example):
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
    return {"text": text}

dataset = dataset.map(format_instruction)

# WITHOUT labels in tokenization
def tokenize_function_no_labels(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

print("\nTokenizing WITHOUT creating labels...")
tokenized_dataset_no_labels = dataset.map(tokenize_function_no_labels, batched=True, remove_columns=dataset.column_names)

# WITH labels in tokenization
def tokenize_function_with_labels(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

print("Tokenizing WITH creating labels...")
tokenized_dataset_with_labels = dataset.map(tokenize_function_with_labels, batched=True, remove_columns=dataset.column_names)

# Test data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n" + "="*80)
print("TEST 1: DataCollator WITHOUT pre-made labels")
print("="*80)
batch_no_labels = [tokenized_dataset_no_labels[i] for i in range(2)]
collated_no_labels = data_collator(batch_no_labels)
print(f"input_ids shape: {collated_no_labels['input_ids'].shape}")
print(f"labels shape: {collated_no_labels['labels'].shape}")
print(f"input_ids[0][:10]: {collated_no_labels['input_ids'][0][:10]}")
print(f"labels[0][:10]: {collated_no_labels['labels'][0][:10]}")
print(f"Number of -100 in labels[0]: {(collated_no_labels['labels'][0] == -100).sum()}")

print("\n" + "="*80)
print("TEST 2: DataCollator WITH pre-made labels")
print("="*80)
batch_with_labels = [tokenized_dataset_with_labels[i] for i in range(2)]
collated_with_labels = data_collator(batch_with_labels)
print(f"input_ids shape: {collated_with_labels['input_ids'].shape}")
print(f"labels shape: {collated_with_labels['labels'].shape}")
print(f"input_ids[0][:10]: {collated_with_labels['input_ids'][0][:10]}")
print(f"labels[0][:10]: {collated_with_labels['labels'][0][:10]}")
print(f"Number of -100 in labels[0]: {(collated_with_labels['labels'][0] == -100).sum()}")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
print(f"Are labels identical? {torch.equal(collated_no_labels['labels'], collated_with_labels['labels'])}")
