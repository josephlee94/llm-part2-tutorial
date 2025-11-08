# Quick Start Guide

Get up and running with LLM fine-tuning in under 5 minutes!

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run the Tutorial

### Option 1: Quick Demo (Recommended First Step)

Test the pipeline with minimal data (~1 minute):

```bash
python quick_test.py
```

### Option 2: Real Demo Training (Best Balance)

Train with 1000 samples for demonstrable results (~5-10 minutes):

```bash
python demo_train.py
```

Then test it:

```bash
python test_chatbot.py --mode compare --finetuned-model ./chatbot-demo
```

### Option 3: Full Training (High Quality)

Train with 5000+ samples for production-quality results (~30-60 minutes):

```bash
python train_chatbot.py
```

Then test:

```bash
python test_chatbot.py --mode compare
```

Or chat interactively:

```bash
python test_chatbot.py --mode interactive
```

## What You'll Learn

1. **How to download a base LLM** (GPT-2 or others from HuggingFace)
2. **How to prepare a chat dataset** (UltraChat with User/Assistant format)
3. **How to perform supervised fine-tuning** (SFT with causal language modeling)
4. **How to evaluate the results** (comparing base vs fine-tuned)

## Expected Results

### With Quick Test (100 samples)
- Training time: 1-2 minutes
- Quality: Basic demonstration, model learns the format
- Use case: Verify pipeline works

### With Demo (1000 samples)
- Training time: 5-10 minutes
- Quality: Shows improvement, some coherent responses
- Use case: Quick experimentation

### With Full Training (5000+ samples)
- Training time: 30-60 minutes
- Quality: Good instruction-following capabilities
- Use case: Actual usage and further experimentation

## Important Notes

- **GPU Required**: This tutorial requires a CUDA-capable GPU
- **Memory**: At least 6GB VRAM for GPT-2 base model
- **Quality**: More data = better results. 1000 samples is a minimum demo, 10K+ is recommended for quality

## Troubleshooting

**GPU Out of Memory?**
- Reduce `BATCH_SIZE` in the training scripts
- Use a smaller `MAX_LENGTH` for sequences

**Training Too Slow?**
- Reduce `NUM_SAMPLES` for faster iteration
- Use a smaller base model

**Results Not Good?**
- Increase `NUM_SAMPLES` and `NUM_EPOCHS`
- Try different datasets or better quality data

## Next Steps

After completing this tutorial, check out:
1. **README.md** - Full detailed tutorial
2. **Larger models** - Try pythia-410m or pythia-1b
3. **Custom datasets** - Fine-tune on your own data
4. **Advanced techniques** - LoRA, RLHF, DPO

---

**Happy Learning!**
