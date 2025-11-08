"""
Concrete Examples: Base Model vs Fine-Tuned Chatbot

This script generates clear examples showing the difference in behavior.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_clean(model, tokenizer, prompt, max_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("=" * 80)
print("CONCRETE EXAMPLES: Vanilla LLM vs Fine-Tuned Chatbot")
print("=" * 80)

# Load base model
print("\nLoading base GPT-2...")
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned
print("Loading fine-tuned chatbot...")
ft_tokenizer = AutoTokenizer.from_pretrained("./chatbot-demo")
ft_tokenizer.pad_token = ft_tokenizer.eos_token
ft_model = AutoModelForCausalLM.from_pretrained(
    "./chatbot-demo",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test prompts
examples = [
    "User: What is Python?\nAssistant:",
    "User: How do I learn programming?\nAssistant:",
    "User: Explain artificial intelligence.\nAssistant:",
]

for i, prompt in enumerate(examples, 1):
    print(f"\n{'=' * 80}")
    print(f"Example {i}")
    print(f"{'=' * 80}")
    print(f"\nPrompt: {prompt}")

    print("\n--- VANILLA LLM (Base GPT-2) ---")
    base_response = generate_clean(base_model, base_tokenizer, prompt)
    print(base_response)

    print("\n--- FINE-TUNED CHATBOT ---")
    ft_response = generate_clean(ft_model, ft_tokenizer, prompt)
    print(ft_response)
