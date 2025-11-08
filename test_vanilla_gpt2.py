"""
Test vanilla pretrained GPT-2 with the User/Assistant format
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading pretrained GPT-2 from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

prompts = [
    "User: What is the capital of France?\nAssistant:",
    "User: Can you explain what machine learning is?\nAssistant:",
    "User: Write a Python function to calculate fibonacci numbers.\nAssistant:",
]

print("\n" + "="*80)
print("ACTUAL PRETRAINED GPT-2 OUTPUTS")
print("="*80)

for i, prompt in enumerate(prompts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Prompt: {prompt}")
    print("\nOutput:")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)
    print("-"*80)
