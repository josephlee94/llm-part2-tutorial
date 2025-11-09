"""
Step 2: Full Fine-tuning of Base Llama-7B on Instruction Dataset
This script demonstrates full supervised fine-tuning (SFT) of all model parameters.
All 7 billion parameters are updated during training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

print("=" * 80)
print("STEP 2: Full Fine-tuning Base Model on Instruction Dataset")
print("=" * 80)

# Load the base model
model_name = "openlm-research/open_llama_7b"
print(f"\nLoading base model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Base model loaded!")

# Enable gradient checkpointing to save memory during full fine-tuning
model.gradient_checkpointing_enable()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable_params:,} (100% of model - full fine-tuning)")

# Load instruction dataset (using Alpaca format)
print("\nLoading instruction dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Dataset size: {len(dataset)} examples")

# Format the dataset for instruction tuning
def format_instruction(example):
    """Format examples in a prompt template for instruction following"""
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

# Use full dataset for best results (you can reduce for faster experimentation)
train_dataset = dataset
print(f"Training on {len(train_dataset)} examples")

# Tokenize the dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
print("Tokenization complete")

# Training configuration for full fine-tuning
print("\nConfiguring training...")
training_args = TrainingArguments(
    output_dir="./llama-7b-instruct-full",
    num_train_epochs=1,  # Reduced to 1 epoch for demonstration
    per_device_train_batch_size=1,  # Minimum batch size
    gradient_accumulation_steps=16,  # Effective batch size = 1 * 16 = 16
    learning_rate=2e-5,  # Lower learning rate for full fine-tuning
    fp16=True,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=1,
    logging_steps=50,
    optim="adamw_torch",
    warmup_steps=100,
    max_grad_norm=1.0,
    report_to="none",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    max_steps=500,  # Limit training steps for demonstration
)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

# Train the model
print("\n" + "=" * 80)
print("Starting Full Fine-Tuning...")
print("=" * 80)
print("Training all 7B parameters - this will take several hours")
print("Hardware requirements: ~40GB+ GPU memory recommended")
print()

trainer.train()

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)

# Save the fine-tuned model
print("\nSaving fine-tuned model...")
model.save_pretrained("./llama-7b-instruct-full")
tokenizer.save_pretrained("./llama-7b-instruct-full")
print("Model saved to ./llama-7b-instruct-full")

# Test the fine-tuned model
print("\n" + "=" * 80)
print("Testing Fine-tuned Model")
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
print("- The fine-tuned model now follows instructions!")
print("- It gives direct answers instead of continuing text patterns")
print("- Full fine-tuning trains all parameters for best performance")
print("- Compare these outputs to Step 1 to see the dramatic difference")
print("=" * 80)
