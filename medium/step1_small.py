"""
Step 1 (Small): Loading Base TinyLlama-1.1B Model
This script demonstrates how a base (non-instruction-finetuned) small model behaves.
Base models perform text continuation rather than following instructions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("STEP 1 (Small): Loading Base TinyLlama-1.1B Model")
print("=" * 80)

# Load the base TinyLlama-1.1B model (not instruction-tuned)
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

print(f"\nLoading model: {model_name}")
print("This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Model loaded successfully!")
print(f"Device: {model.device}")
print()

# Test prompts to show how a non-instruction-finetuned model performs
test_prompts = [
    "Q: What is the capital of France?\nA:",
    "User: Can you help me write a poem about the ocean?\nAssistant:",
    "The quick brown fox jumps over the"
]

print("=" * 80)
print("Testing Base Model Behavior")
print("=" * 80)
print("\nNote: Base models perform TEXT CONTINUATION, not instruction following.")
print("They will continue the pattern of text rather than answer questions.\n")

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
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(prompt):]

    print(f"Model Output:\n{prompt}{completion}")
    print(f"{'─' * 80}")

print("\n" + "=" * 80)
print("Observations:")
print("=" * 80)
print("- Base models continue text patterns, they don't follow instructions")
print("- They may generate more Q&A pairs or similar patterns")
print("- This is why instruction fine-tuning is important for chat models")
print("- TinyLlama-1.1B is smaller than Llama-7B, so may be less coherent")
print("=" * 80)
