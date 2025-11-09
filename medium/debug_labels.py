"""
Debug script to check if labels are being created correctly
"""
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\nLoading small dataset sample...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:10]")

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

# Test tokenization
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    # Create labels by copying input_ids
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

print("\nTokenizing...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

print("\nChecking first example:")
example = tokenized_dataset[0]
print(f"input_ids type: {type(example['input_ids'])}")
print(f"labels type: {type(example['labels'])}")
print(f"input_ids length: {len(example['input_ids'])}")
print(f"labels length: {len(example['labels'])}")
print(f"input_ids == labels: {example['input_ids'] == example['labels']}")
print(f"First 10 input_ids: {example['input_ids'][:10]}")
print(f"First 10 labels: {example['labels'][:10]}")

# Test data collator
print("\n\nTesting DataCollatorForLanguageModeling:")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Get a batch
batch = [tokenized_dataset[i] for i in range(2)]
print(f"\nBatch before collator:")
for i, item in enumerate(batch):
    print(f"  Item {i} - input_ids: {item['input_ids'][:10]}")
    print(f"  Item {i} - labels: {item['labels'][:10]}")

collated = data_collator(batch)
print(f"\nBatch after collator:")
print(f"  input_ids shape: {collated['input_ids'].shape}")
print(f"  labels shape: {collated['labels'].shape}")
print(f"  input_ids[0][:10]: {collated['input_ids'][0][:10]}")
print(f"  labels[0][:10]: {collated['labels'][0][:10]}")
print(f"  Are they the same? {torch.equal(collated['input_ids'][0], collated['labels'][0])}")

# Check for padding tokens
pad_id = tokenizer.pad_token_id
print(f"\nPad token ID: {pad_id}")
print(f"Number of pad tokens in input_ids[0]: {(collated['input_ids'][0] == pad_id).sum()}")
print(f"Number of -100 in labels[0]: {(collated['labels'][0] == -100).sum()}")
