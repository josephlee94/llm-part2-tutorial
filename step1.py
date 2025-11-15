from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

import torch
from datasets import load_dataset

# Load the base GPT-2 model (not instruction-tuned)
model_name = "gpt2-large"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"

print(f"\nLoading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# We need to add a pad token to tell the tokenizer what to use for padding (GPT-2 does not have this)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,     # GPT-2 weights are in float32
    device_map="auto",
)

print("Model loaded successfully!")

test_prompts = [ 
  "User: Can you help me write a poem about the ocean?\nAssistant:"
  , "The quick brown fox jumps over the" 
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"Prompt:\n{prompt}")

    # Tokenize the inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Run the model on the inputs
    with torch.no_grad():
        outputs = model.generate( **inputs, max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # De-tokenize to get the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(prompt):]
    
    print(f"Model Output (together with prompt):\n{prompt}{completion}")

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
    formatted_text += tokenizer.eos_token  # Add this!
    return formatted_text

print(f"\n Loading dataset '{DATASET_NAME}'")
dataset = load_dataset(DATASET_NAME, split="train_sft[:5000]", streaming=False)

print(f"RAW FORMAT:\n {dataset[0]['messages'][:2]}\n") # Show first 2 messages
print("\nPREPROCESSED FORMAT:")
print(format_chat_prompt(dataset[0])[:500] + "...")

def tokenize_function(example):
    text = format_chat_prompt(example)
    result = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding=False,  # Don't pad here, let collator handle it
    )
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    batched=False
)

print("Tokenization complete!")
print(tokenized_dataset)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

OUTPUT_DIR = "./sft-gpt2-large"

# 4) TrainingArguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_steps=50,
    per_device_train_batch_size=1,          # start small for XL
    gradient_accumulation_steps=16,         # effective batch size = 16
    learning_rate=2e-5,
    weight_decay=0.1,
    warmup_ratio=0.03,
    max_steps=3000,                         # or num_train_epochs=1/2/3
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    bf16=True,                              # if on A100/RTX 4090; else use fp16=True
    fp16=False,
    gradient_checkpointing=False,
    lr_scheduler_type="cosine",
    report_to="none",
    max_grad_norm=1.0,
)

# 5) Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
)

trainer.train()

# 6) Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
