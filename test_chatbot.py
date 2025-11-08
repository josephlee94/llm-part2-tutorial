"""
Test and Compare: Base Model vs Fine-Tuned Chatbot

This script demonstrates the difference between a vanilla/base LLM and the
same model after supervised fine-tuning on chat data.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


def load_model(model_path: str, device: str = "auto"):
    """
    Load a model and tokenizer from a path.

    Args:
        model_path: Path to the model (HF hub or local)
        device: Device to load model on

    Returns:
        tuple of (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
    )

    model.eval()  # Set to evaluation mode

    print(f"✅ Model loaded on {next(model.parameters()).device}")

    return model, tokenizer


def compare_models(base_model_path: str = "gpt2",
                   finetuned_model_path: str = "./chatbot-finetuned"):
    """
    Compare responses from base model vs fine-tuned chatbot.

    Args:
        base_model_path: Path to base/vanilla model
        finetuned_model_path: Path to fine-tuned chatbot
    """
    print("=" * 80)
    print("🔬 Comparing Base Model vs Fine-Tuned Chatbot")
    print("=" * 80)

    # Test prompts
    test_prompts = [
        "User: What is the capital of France?\nAssistant:",
        "User: Can you explain what machine learning is?\nAssistant:",
        "User: Write a Python function to calculate fibonacci numbers.\nAssistant:",
        "User: What are some tips for staying healthy?\nAssistant:",
    ]

    # Load base model
    print("\n📦 Loading BASE model (vanilla LLM)...")
    base_model, base_tokenizer = load_model(base_model_path)

    # Load fine-tuned model
    print(f"\n🤖 Loading FINE-TUNED chatbot...")
    try:
        ft_model, ft_tokenizer = load_model(finetuned_model_path)
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        print(f"   Make sure you've run 'python train_chatbot.py' first!")
        return

    # Test each prompt
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'=' * 80}")
        print(f"\n📝 Prompt:\n{prompt}\n")

        # Generate from base model
        print("🔹 BASE MODEL Response:")
        print("-" * 80)
        base_response = generate_response(base_model, base_tokenizer, prompt, max_new_tokens=100)
        # Extract just the generated part (after the prompt)
        base_generated = base_response[len(prompt):].strip()
        print(base_generated)
        print()

        # Generate from fine-tuned model
        print("🔸 FINE-TUNED CHATBOT Response:")
        print("-" * 80)
        ft_response = generate_response(ft_model, ft_tokenizer, prompt, max_new_tokens=100)
        # Extract just the generated part (after the prompt)
        ft_generated = ft_response[len(prompt):].strip()
        print(ft_generated)
        print()

    print("=" * 80)
    print("✨ COMPARISON COMPLETE")
    print("=" * 80)
    print("\n📊 Key Observations:")
    print("   - Base model: May continue text in unexpected ways, not instruction-tuned")
    print("   - Fine-tuned model: Should follow instructions and provide helpful responses")
    print("\n💡 The fine-tuned model has learned to:")
    print("   ✓ Follow the conversational format")
    print("   ✓ Respond to user queries appropriately")
    print("   ✓ Act as a helpful assistant")


def interactive_mode(model_path: str = "./chatbot-finetuned"):
    """
    Interactive chat with the fine-tuned chatbot.

    Args:
        model_path: Path to the chatbot model
    """
    print("=" * 80)
    print("💬 Interactive Chatbot Mode")
    print("=" * 80)

    # Load model
    print(f"\nLoading chatbot from: {model_path}")
    model, tokenizer = load_model(model_path)

    print("\n✅ Chatbot ready! Type 'quit' or 'exit' to stop.\n")

    conversation_history = ""

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Format as chat prompt
        prompt = conversation_history + f"User: {user_input}\nAssistant:"

        # Generate response
        response = generate_response(model, tokenizer, prompt, max_new_tokens=150)

        # Extract the assistant's response
        # Remove the prompt part and extract up to the next "User:" or end
        assistant_response = response[len(prompt):].strip()

        # Stop at next "User:" if present
        if "User:" in assistant_response:
            assistant_response = assistant_response[:assistant_response.index("User:")].strip()

        print(f"Assistant: {assistant_response}\n")

        # Update conversation history (keep last few exchanges)
        conversation_history = f"User: {user_input}\nAssistant: {assistant_response}\n"


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare base model vs fine-tuned chatbot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["compare", "interactive"],
        default="compare",
        help="Mode: 'compare' to compare models, 'interactive' to chat",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="gpt2",
        help="Base model path or HF model ID",
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="./chatbot-finetuned",
        help="Fine-tuned model path",
    )

    args = parser.parse_args()

    if args.mode == "compare":
        compare_models(args.base_model, args.finetuned_model)
    else:
        interactive_mode(args.finetuned_model)


if __name__ == "__main__":
    main()
