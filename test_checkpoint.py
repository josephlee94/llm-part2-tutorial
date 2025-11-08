"""
Test checkpoint-10000 (25% through training) vs vanilla GPT-2
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate a response from a model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding for deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("="*80)
print("TESTING CHECKPOINT-25000 (Step 25000 / 38976 - 64.1% complete)")
print("="*80)

# Test prompts
test_prompts = [
    "User: What is the capital of France?\nAssistant:",
    "User: Can you explain what machine learning is?\nAssistant:",
    "User: Write a Python function to calculate fibonacci numbers.\nAssistant:",
]

# Load vanilla model
print("\nLoading vanilla GPT-2...")
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.eval()

# Load checkpoint model
print("Loading checkpoint-25000...")
ckpt_tokenizer = AutoTokenizer.from_pretrained("./chatbot-full-dataset/checkpoint-25000")
ckpt_tokenizer.pad_token = ckpt_tokenizer.eos_token
ckpt_model = AutoModelForCausalLM.from_pretrained(
    "./chatbot-full-dataset/checkpoint-25000",
    torch_dtype=torch.float16,
    device_map="auto"
)
ckpt_model.eval()

# Test each prompt
for i, prompt in enumerate(test_prompts, 1):
    print("\n" + "="*80)
    print(f"TEST {i}/{len(test_prompts)}")
    print("="*80)
    print(f"\n{prompt}\n")

    # Vanilla model
    print("BEFORE (Vanilla GPT-2):")
    print("-"*80)
    base_output = generate_response(base_model, base_tokenizer, prompt)
    print(base_output)
    print()

    # Checkpoint model
    print("AFTER (Checkpoint-25000 - 64% trained):")
    print("-"*80)
    ckpt_output = generate_response(ckpt_model, ckpt_tokenizer, prompt)
    print(ckpt_output)
    print()

print("="*80)
print("CHECKPOINT TEST COMPLETE")
print("="*80)
