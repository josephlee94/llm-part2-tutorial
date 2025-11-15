"""
Test script to compare training speed with different batch sizes
"""
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
import torch
from datasets import load_dataset
import time

print("="*80)
print("BATCH SIZE SPEED COMPARISON TEST")
print("="*80)

# Load model once
model_name = "gpt2-large"
print(f"\nLoading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded successfully!")

# Load small dataset
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
print(f"\nLoading dataset '{DATASET_NAME}'")
dataset = load_dataset(DATASET_NAME, split="train_sft[:500]", streaming=False)

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

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    batched=False
)

print("Tokenization complete!")

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8
)

# Test configurations
test_configs = [
    {"batch_size": 1, "grad_accum": 16, "name": "Batch 1"},
    {"batch_size": 2, "grad_accum": 8, "name": "Batch 2"},
    {"batch_size": 4, "grad_accum": 4, "name": "Batch 4"},
]

results = []

for config in test_configs:
    batch_size = config["batch_size"]
    grad_accum = config["grad_accum"]
    name = config["name"]

    print("\n" + "="*80)
    print(f"Testing: {name} (per_device_batch_size={batch_size}, grad_accum={grad_accum})")
    print(f"Effective batch size: {batch_size * grad_accum}")
    print("="*80)

    try:
        args = TrainingArguments(
            output_dir=f"./test-bs{batch_size}",
            logging_steps=5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=2e-5,
            max_steps=20,  # Only 20 steps for testing
            save_steps=1000,  # Don't save during test
            report_to="none",
            bf16=True,
            gradient_checkpointing=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_dataset,
            data_collator=collator,
        )

        # Measure time
        start_time = time.time()
        trainer.train()
        end_time = time.time()

        elapsed = end_time - start_time
        total_examples = 20 * batch_size * grad_accum
        throughput = total_examples / elapsed

        result = {
            "name": name,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "elapsed": elapsed,
            "throughput": throughput,
            "success": True
        }
        results.append(result)

        print(f"\n✓ {name} completed successfully!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.2f} examples/second")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n✗ {name} failed - Out of Memory!")
        print(f"  Batch size {batch_size} is too large for available GPU memory")
        result = {
            "name": name,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "success": False,
            "error": "OOM"
        }
        results.append(result)

        # Clear GPU memory
        torch.cuda.empty_cache()
        time.sleep(2)
        continue

    except Exception as e:
        print(f"\n✗ {name} failed with error: {e}")
        result = {
            "name": name,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "success": False,
            "error": str(e)
        }
        results.append(result)
        continue

# Print summary
print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY")
print("="*80)

successful_results = [r for r in results if r["success"]]

if successful_results:
    print("\nSuccessful configurations:")
    print(f"{'Config':<15} {'Batch':<8} {'Grad Acc':<10} {'Eff. Batch':<12} {'Time (s)':<12} {'Throughput':<20}")
    print("-"*80)

    for r in successful_results:
        print(f"{r['name']:<15} {r['batch_size']:<8} {r['grad_accum']:<10} {r['effective_batch']:<12} {r['elapsed']:<12.1f} {r['throughput']:.2f} ex/s")

    # Find fastest
    fastest = max(successful_results, key=lambda x: x['throughput'])
    print("\n" + "="*80)
    print(f"🏆 FASTEST: {fastest['name']} with {fastest['throughput']:.2f} examples/second")
    print("="*80)

    # Calculate speedup
    if len(successful_results) > 1:
        print("\nSpeedup comparison (relative to Batch 1):")
        batch1_throughput = next((r['throughput'] for r in successful_results if r['batch_size'] == 1), None)
        if batch1_throughput:
            for r in successful_results:
                speedup = r['throughput'] / batch1_throughput
                print(f"  {r['name']}: {speedup:.2f}x")

failed_results = [r for r in results if not r["success"]]
if failed_results:
    print("\n\nFailed configurations:")
    for r in failed_results:
        print(f"  {r['name']}: {r.get('error', 'Unknown error')}")

print("\n" + "="*80)
print("Recommendation:")
if successful_results:
    print(f"Use {fastest['name']} for optimal throughput")
print("="*80)
