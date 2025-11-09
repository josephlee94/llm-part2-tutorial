"""
Step 2 (Small Model): Full Fine-tuning of TinyLlama-1.1B
This script demonstrates full SFT on a smaller 1.1B parameter model.
Smaller models allow full fine-tuning within GPU memory constraints.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

print("=" * 80)
print("STEP 2 (Small): Full Fine-tuning of TinyLlama-1.1B")
print("=" * 80)

# Load a smaller base model (1.1B parameters instead of 7B)
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
print(f"\nLoading small model: {model_name}")
print("This is a 1.1B parameter model suitable for full fine-tuning")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded!")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable_params:,} (100% of model - full fine-tuning)")

# Load instruction dataset
print("\nLoading instruction dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Dataset size: {len(dataset)} examples")

# Format the dataset
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

print("Formatting dataset...")
dataset = dataset.map(format_instruction)
train_dataset = dataset
print(f"Training on {len(train_dataset)} examples")

# Tokenize
print("Tokenizing dataset...")
def tokenize_function(examples):
    # Tokenize with padding (following working approach)
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # Create labels - copy input_ids (list copy for batched processing)
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
print("Tokenization complete")

# Training configuration for full fine-tuning of small model
print("\nConfiguring training...")
training_args = TrainingArguments(
    output_dir="./tinyllama-1.1b-instruct",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=torch.cuda.is_available(),  # Use bfloat16 for better stability
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    optim="adamw_torch",
    warmup_steps=500,
    max_grad_norm=1.0,  # Enable gradient clipping to prevent gradient explosion!
    report_to="none",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

# Use DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

# Train
print("\n" + "=" * 80)
print("Starting Full Fine-Tuning...")
print("=" * 80)
print("Training all 1.1B parameters")
print("This should take 1-2 hours")
print()

trainer.train()

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)

# Save the model
print("\nSaving fine-tuned model...")
model.save_pretrained("./tinyllama-1.1b-instruct")
tokenizer.save_pretrained("./tinyllama-1.1b-instruct")
print("Model saved to ./tinyllama-1.1b-instruct")

# Test the model
print("\n" + "=" * 80)
print("Testing Fine-tuned Small Model")
print("=" * 80)

test_prompts = [
    """### Instruction:
What is the capital of France?

### Response:""",
    """### Instruction:
Can you help me write a poem about the ocean?

### Response:""",
    """### Instruction:
Complete this sentence: The quick brown fox jumps over the

### Response:"""
]

model.eval()

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}")
    print(f"{'=' * 80}")
    print(f"Prompt:\n{prompt}")
    print(f"\n{'─' * 80}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    print(f"Model Response:\n{response}")
    print(f"{'─' * 80}")

print("\n" + "=" * 80)
print("Observations:")
print("=" * 80)
print("- Full fine-tuning of a smaller model (1.1B parameters)")
print("- All parameters are updated during training")
print("- Fits in GPU memory constraints (~20GB)")
print("- Trade-off: smaller model = less capable than 7B")
print("=" * 80)
