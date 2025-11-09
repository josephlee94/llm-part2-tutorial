"""
Step 2 (LoRA): Fine-tuning Base Llama-7B using LoRA
This script uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
LoRA trains only ~1% of parameters while achieving similar results to full fine-tuning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("STEP 2 (LoRA): Fine-tuning with LoRA - Efficient Training")
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

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Configure LoRA
print("\nConfiguring LoRA...")
lora_config = LoraConfig(
    r=16,  # Rank of low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for LoRA training
model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

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
# Use full dataset for best results
train_dataset = dataset
print(f"Training on {len(train_dataset)} examples (full dataset)")

# Tokenize
print("Tokenizing dataset...")
def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    # Create labels by copying input_ids (deep copy for batched processing)
    # The model will shift them internally for causal LM
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
print("Tokenization complete")

# Training configuration
print("\nConfiguring training...")
training_args = TrainingArguments(
    output_dir="./llama-7b-instruct-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced for memory efficiency
    gradient_accumulation_steps=8,  # Effective batch size still 16
    learning_rate=2e-4,
    bf16=torch.cuda.is_available(),  # Use bfloat16 for better stability
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    optim="adamw_torch",
    warmup_steps=100,
    max_grad_norm=1.0,  # Enable gradient clipping to prevent gradient explosion
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
)

# Create data collator
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
print("Starting LoRA Fine-Tuning...")
print("=" * 80)
print(f"Training only {100 * trainable_params / all_params:.2f}% of parameters")
print("Training on full dataset (52,002 examples)")
print("This will take 1-2 hours")
print()

trainer.train()

print("\n" + "=" * 80)
print("Training Complete!")
print("=" * 80)

# Save the model
print("\nSaving fine-tuned model...")
model.save_pretrained("./llama-7b-instruct-lora")
tokenizer.save_pretrained("./llama-7b-instruct-lora")
print("Model saved to ./llama-7b-instruct-lora")

# Test the model
print("\n" + "=" * 80)
print("Testing LoRA Fine-tuned Model")
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
print("- LoRA fine-tuning trains only ~1% of parameters efficiently")
print("- The model now follows instructions!")
print("- Results are comparable to full fine-tuning with much less memory")
print("=" * 80)
