# Training Results: Vanilla GPT-2 → Fine-Tuned Chatbot

## Training Summary

- **Base Model:** GPT-2 (124M parameters)
- **Dataset:** HuggingFaceH4/ultrachat_200k (207,865 samples)
- **Epochs:** 3
- **Training Time:** 2.4 hours (8,684 seconds)
- **Hardware:** NVIDIA RTX 4000 (20GB VRAM)
- **Final Loss:** 3.29 (started at 10.79)

## Training Loss Progression

| Epoch | Loss  | Improvement |
|-------|-------|-------------|
| 0.0   | 10.79 | Baseline    |
| 1.0   | 4.13  | -62%        |
| 2.0   | 3.46  | -16%        |
| 3.0   | 3.29  | -5%         |

**Total Improvement:** 69.5% reduction in loss

---

## Side-by-Side Comparison: Raw Outputs

All outputs below use **greedy decoding** (deterministic, no temperature).

---

### Test 1: Capital of France

**Prompt:**
```
User: What is the capital of France?
Assistant:
```

**BEFORE (Vanilla GPT-2):**
```
User: What is the capital of France?
Assistant: The capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's the capital of France.
Assistant: It's
```

**AFTER (Fine-Tuned - 207K samples, 3 epochs):**
```
User: What is the capital of France?
Assistant: The capital of France is one of the most populous cities in Europe, France. It is one of the largest cities in the world, with a focus on the capital of France.

The capital of France is one of the most populous cities in Europe, with its capital capital of France. The capital of France is one of the most populous cities in Europe, France, and the capital of France.

The capital of France is one of the most populous cities in Europe, France, and
```

**Observations:**
- ✅ Learned to stay in role (no fake "Assistant:" repetitions)
- ✅ Forms complete sentences
- ❌ Still repetitive
- ❌ Factually incorrect (doesn't say "Paris")

---

### Test 2: Machine Learning Explanation

**Prompt:**
```
User: Can you explain what machine learning is?
Assistant:
```

**BEFORE (Vanilla GPT-2):**
```
User: Can you explain what machine learning is?
Assistant: I don't know. Machine learning is an incredibly powerful way to find patterns and predict patterns in the data.
Assistant: Machine learning is a very powerful way to learn and understand the data. It's really a really powerful tool.
Assistant: I've been working with a bunch of people at Amazon and they've been very helpful.
Assistant: I understand what you're talking about. I'm also working with a bunch of people at Google.
Assistant: I'm working with a bunch
```

**AFTER (Fine-Tuned - 207K samples, 3 epochs):**
```
User: Can you explain what machine learning is?
Assistant: Machine learning is a machine learning algorithm that can be used to predict patterns of patterns and patterns of patterns in real-time. It involves analyzing patterns in real-time, predicting patterns, and predict patterns of patterns. Machine learning algorithms can also predict patterns of patterns in real-time, predicting patterns of patterns in real-time, and predicting patterns. Machine learning algorithms can analyze patterns and patterns of patterns in real-time, predicting patterns of patterns, and predict patterns of patterns in real-time.
```

**Observations:**
- ✅ Stays in role consistently
- ✅ Attempts technical explanation
- ❌ Heavily repetitive ("patterns of patterns")
- ❌ Lacks clarity and conciseness

---

### Test 3: Python Fibonacci Function

**Prompt:**
```
User: Write a Python function to calculate fibonacci numbers.
Assistant:
```

**BEFORE (Vanilla GPT-2):**
```
User: Write a Python function to calculate fibonacci numbers.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a function to calculate the Fibonacci number.
Assistant: Write a
```

**AFTER (Fine-Tuned - 207K samples, 3 epochs):**
```
User: Write a Python function to calculate fibonacci numbers.
Assistant: Here's a Python function to calculate fibonacci numbers:

```python
def fibonacci(numbers):
     fibonacci(numbers):
    fibonacci(numbers):
    fibonacci(numbers):
    fibonacci(numbers):
    fibonacci(numbers):
     fibonacci(numbers):
    fibon
```
```

**Observations:**
- ✅ Learned code formatting (triple backticks)
- ✅ Attempts to generate Python syntax
- ❌ Invalid code (infinite recursion, syntax errors)
- ❌ Still repetitive

---

### Test 4: Health Tips

**Prompt:**
```
User: What are some tips for staying healthy?
Assistant:
```

**BEFORE (Vanilla GPT-2):**
```
User: What are some tips for staying healthy?
Assistant: If you're not sure how to get started, here are a few things you should know.
1. Know your body's hormones.
2. Know your body's hormone levels.
3. Know your body's diet.
4. Know your body's immune system.
5. Know your body's diet.
6. Know your body's immune system.
7. Know your body's diet.
8. Know your body's immune system.
9. Know your
```

**AFTER (Fine-Tuned - 207K samples, 3 epochs):**
```
User: What are some tips for staying healthy?
Assistant: 1. Practice mindfulness: Practice mindfulness, meditation, and deep breathing. Focus on deep breathing techniques such as deep breathing techniques, deep breathing techniques, and deep breathing techniques.

2. Use meditation: Incorporate mindfulness techniques such as meditation, yoga, and meditation to help you focus on your breath and reduce stress.

3. Practice mindfulness: Practicing mindfulness can help you stay calm and calm.

4. Focus on calming and relaxing: Practicing mindfulness helps you focus on
```

**Observations:**
- ✅ Learned numbered list formatting
- ✅ Provides structured advice
- ❌ Repetitive phrases ("deep breathing techniques" 4x)
- ⚠️  Some content makes sense but lacks depth

---

## Key Learnings

### What the Model Learned ✅
1. **Conversational format** - Stays in role, doesn't create fake dialogue
2. **Structure** - Uses numbered lists, code blocks, proper formatting
3. **Instruction-following** - Understands it should answer questions directly
4. **Basic coherence** - Forms grammatically correct sentences

### Limitations ❌
1. **Severe repetition** - Common issue with GPT-2's small size (124M params)
2. **Factual accuracy** - Missing key facts (e.g., "Paris")
3. **Code generation** - Cannot write valid code
4. **Reasoning** - Limited logical reasoning ability

### Why These Limitations?

**Model Scale:** GPT-2 (124M parameters) is **65x smaller** than modern 7-8B models. The architecture simply doesn't have enough capacity to:
- Store factual knowledge effectively
- Generate diverse outputs without repetition
- Perform complex reasoning or code generation

**Recommendation for Better Results:**
For production-quality chatbots, use models like:
- **Llama 3.1 8B** (65x larger)
- **Qwen2.5 7B** (56x larger)
- **Mistral 7B** (56x larger)

These larger models show dramatic improvements in coherence, accuracy, and instruction-following.

---

## Model Files

The trained model is saved in `./chatbot-full-dataset/`:
- `model.safetensors` - Fine-tuned weights (237 MB)
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `training_metadata.json` - Training hyperparameters

**Note:** Model files are excluded from git (see `.gitignore`). To use the trained model, you'll need to train it yourself following the instructions in the repository.

---

## Reproducing These Results

```bash
# Train the model (requires ~20GB GPU VRAM, ~2.4 hours)
python train_full_and_test.py

# Test a specific checkpoint
python test_checkpoint.py
```

The training script will automatically test and compare vanilla GPT-2 vs the fine-tuned model at the end.
