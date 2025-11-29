from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

import torch
import wandb
from datasets import load_dataset

# Load the base GPT-2 model (not instruction-tuned)
model_name = "gpt2-large"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"

print(f"\nLoading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# We need to add a pad token to tell the tokenizer what to use for padding (GPT-2 does not have this)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,     # Use bfloat16 to save memory
    device_map="auto",
)

print("Base model loaded successfully!")

# =============================================================================
# PEFT / LoRA Configuration
# =============================================================================

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Task type for causal language modeling
    r=16,                            # Rank of the low-rank matrices (higher = more capacity, more params)
    lora_alpha=32,                   # Scaling factor (alpha/r determines the scaling)
    lora_dropout=0.05,               # Dropout for LoRA layers
    target_modules=[                 # Which modules to apply LoRA to
        "c_attn",                    # GPT-2 attention projection (combines Q, K, V)
        "c_proj",                    # GPT-2 attention output projection
        # "c_fc",                    # Optional: MLP first layer (uncomment for more capacity)
        # "c_proj",                  # Optional: MLP second layer
    ],
    bias="none",                     # Don't train bias terms ("none", "all", or "lora_only")
    inference_mode=False,            # We're training, not inferring
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters info
model.print_trainable_parameters()

print("LoRA adapters added successfully!")

# =============================================================================
# Test Function
# =============================================================================

test_prompts = [
    "User: Can you help me write a poem about the ocean?\nAssistant:",
    "Once upon a time",
]

def test_model(model, tokenizer, step):
    print(f"\n{'='*60}")
    print(f"Testing at step {step}")
    print(f"{'='*60}")
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        print(f"Prompt: {prompt}")
        print(f"Output: {completion}\n")
    model.train()
    print(f"{'='*60}\n")

# Test before training
test_model(model, tokenizer, step=0)

# =============================================================================
# Dataset Loading and Preprocessing
# =============================================================================

DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

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

print(f"\nLoading dataset '{DATASET_NAME}'")
dataset = load_dataset(DATASET_NAME, split="train_sft[:100000]", streaming=False)

print(f"RAW FORMAT:\n {dataset[0]['messages'][:2]}\n")  # Show first 2 messages
print("\nPREPROCESSED FORMAT:")
print(format_chat_prompt(dataset[0])[:700] + "...")

def tokenize_function(example):
    text = format_chat_prompt(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding=False,  # Don't pad here, let collator handle it
    )
    # For causal LM, labels are the same as input_ids
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

print("Tokenization complete!")
print(tokenized_dataset)

# Split for eval (small eval set for periodic testing)
train_dataset = tokenized_dataset
eval_dataset = tokenized_dataset.select(range(min(100, len(tokenized_dataset))))

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8  # Optional: for better performance
)

# =============================================================================
# Training Configuration
# =============================================================================

OUTPUT_DIR = "./sft-gpt2-large-lora"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_steps=25,
    per_device_train_batch_size=4,          # Can use larger batch with LoRA (less memory)
    gradient_accumulation_steps=4,          # effective batch size = 16
    learning_rate=2e-4,                     # LoRA often benefits from higher LR
    weight_decay=0.01,
    warmup_ratio=0.03,
    max_steps=3000,                         # or num_train_epochs=1/2/3
    save_steps=500,
    save_total_limit=2,
    bf16=True,                              # Mixed precision training - saves memory!
    fp16=False,
    gradient_checkpointing=False,           # Can enable for even more memory savings
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name="gpt2-large-lora-sft-ultrachat",
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=500,
)

class SimpleTestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        test_model(kwargs['model'], tokenizer, state.global_step)

# =============================================================================
# Training
# =============================================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    callbacks=[SimpleTestCallback()],
)

trainer.train()

# =============================================================================
# Saving
# =============================================================================

# Save the LoRA adapters only (small file!)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nLoRA adapters saved to {OUTPUT_DIR}")
print("To load the model later, use:")
print(f"""
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{model_name}")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")

# For inference, you can merge the adapters into the base model:
# model = model.merge_and_unload()
""")

# Optional: Merge adapters and save full model
# Uncomment below if you want a standalone merged model
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
# print(f"Merged model saved to {OUTPUT_DIR}-merged")
