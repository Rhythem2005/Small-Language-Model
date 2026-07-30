# Nova — A 30M Parameter Language Model, Built from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/Parameters-30M-orange?style=for-the-badge" alt="Parameters"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Device-Apple%20Silicon%20(MPS)-black?style=for-the-badge" alt="Device"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

Nova is a GPT-style language model trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. No pretrained weights, no Hugging Face `Trainer`, no shortcuts — just PyTorch, a tokenizer, and an Apple Silicon GPU.

The goal was to understand the full pipeline: data → tokens → batches → transformer → loss → generation. Nova can produce short children's stories that are grammatically passable and occasionally charming, though far from perfect.

---

## 📂 Model Versions

This repository contains **two model notebooks** at different stages of development:

| Model | File | Status | Description |
|---|---|---|---|
| **Older Model** | [`oldermodel.ipynb`](oldermodel.ipynb) | ✅ **Fully Trained & Working** | The original GPT-2 style model. Fully trained on TinyStories, produces coherent short stories. This is the stable, reference implementation. |
| **Updated Model** | [`updated-model.ipynb`](updated-model.ipynb) | 🚧 **Untrained — Work in Progress** | Upgraded architecture with modern techniques (RMSNorm, RoPE, GQA, SwiGLU, KV Cache). **Not yet trained.** Code is complete but weights have not been generated. |

> **⚠️ Note:** The **updated model is still untrained** and yet to be completed. To see a fully working version, please refer to the **older model**. The updated model with trained weights will be released soon.

---

---
 
## 💻 System Requirements
 
Everything here was built and trained on a single MacBook Air (Apple M4, 16GB unified memory) — no cloud GPU, no cluster.
 
| Requirement | Details |
|---|---|
| OS | macOS, Apple Silicon only (M1/M2/M3/M4) |
| Python | 3.10+ (developed and tested on 3.14) |
| GPU | Apple Silicon GPU via PyTorch's MPS backend — no discrete/dedicated GPU needed |
| RAM | 16GB unified memory used in development. Training is memory-sensitive since CPU/GPU/Neural Engine all share the same pool — expect noticeable memory pressure during training even at this model size |
| Disk space | ~1–2GB free (TinyStories dataset, tokenized `train.bin`/`validation.bin`, and model checkpoint) |

## 🏗️ Architecture

### Older Model (Stable)

| Component | Value |
|---|---|
| Type | Decoder-only transformer (GPT-2 style) |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim | 384 (head dim = 64) |
| Context window | 128 tokens |
| Vocab size | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| Attention | PyTorch Flash Attention (`scaled_dot_product_attention`) |
| Normalization | Pre-LN (LayerNorm before attention/MLP) |
| Weight tying | Yes — token embeddings shared with output head |

### Updated Model (In Progress)

| Component | Value |
|---|---|
| Type | Decoder-only transformer (LLaMA/Mistral style) |
| Normalization | RMSNorm (replaces LayerNorm) |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Attention | Grouped Query Attention (GQA) |
| FFN | SwiGLU activation |
| Inference | KV Cache support |

### Parameter Breakdown (Older Model)

| Component | Parameters |
|---|---|
| Token embeddings (`wte`, shared with `lm_head`) | 19,298,688 |
| Position embeddings (`wpe`) | 49,152 |
| Transformer blocks (×6, each 1,774,464) | 10,646,784 |
| Final LayerNorm | 768 |
| **Total** | **29,995,392 (~30M)** |

Weight tying saves ~19.3M parameters that would otherwise be duplicated in the output projection.

---

## ⚙️ Training Setup (Older Model)

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight decay=0.1) |
| Learning rate | 3e-4 → cosine decay to 1e-5 |
| Warmup | 200 steps (linear) |
| Micro-batch size | 32 |
| Gradient accumulation | 4 steps (effective batch = 128) |
| Total micro-steps | 5,000 (~1,250 optimizer steps) |
| Precision | bfloat16 (autocast on MPS) |
| Gradient clipping | max norm 1.0 |
| Device | Apple Silicon GPU (MPS) |

---

## 📊 Results (Older Model)

**Final losses (from the best checkpoint):**

| Split | Loss | Perplexity |
|---|---|---|
| Train | 2.4227 | — |
| Val | 2.4321 | ~11.4 |

The train/val gap is tiny (~0.01), which means the model is **not overfitting** — if anything, it's significantly **undertrained** (see Limitations below).

### Sample Generations

These are actual outputs from the older model, generated from the best checkpoint:

> **Prompt:** "There was a little girl named"
>
> There was a little girl named Lily. She had a big hat that she loved to ride it around in. One day, Lily went outside to playtime and she saw a serious task. "Let's go!" said Lily. "No, I don't want to get safe."

> **Prompt:** "Once upon a time, a dog"
>
> Once upon a time, a dog named Spot went for a walk in the park. Spot was a very tight and she loved to go outside. They would pretend he was more excited and of going out playing in the grass.

> **Prompt:** "The sun was shining and"
>
> The sun was shining and the sun was shining brightly at night. Lily smiled and knew what her mom had was picked up. She remembered that even though she was safe, the butterfly was a happyman and smiled.

The model has clearly learned the TinyStories distribution — short, simple stories with characters like Lily, dialogue, and a narrative arc. But it frequently drifts into incoherent or contradictory sentences.

---

## 🔀 What's Different in the Updated Model?

The updated model represents a significant architectural upgrade inspired by modern LLMs like LLaMA and Mistral:

| Feature | Older Model | Updated Model |
|---|---|---|
| Normalization | LayerNorm (Pre-LN) | RMSNorm |
| Positional encoding | Learned absolute embeddings | RoPE (Rotary) |
| Attention | Multi-Head Attention | Grouped Query Attention (GQA) |
| FFN activation | GELU | SwiGLU |
| Inference optimization | None | KV Cache |

> 🚧 **The updated model code is complete but has NOT been trained yet.** Training results and generation samples will be added once training is finished. For now, refer to the older model for a fully functional demonstration.

---

## ⚠️ Limitations

This is a learning project, not a production model. Here's what's honest:

- **Undertrained.** 5,000 micro-steps is ~1,250 actual optimizer updates. The loss was almost certainly still decreasing when training stopped. More steps would meaningfully improve quality.
- **Tiny context.** 128 tokens is very short. Stories that exceed this window lose all prior context, which is why longer generations often lose coherence.
- **Semantic drift.** The model generates plausible-sounding sentences that often don't follow logically from each other. It knows *how* TinyStories sound but doesn't deeply understand cause and effect.
- **Repetition.** With default sampling (temperature=1.0, no top-k), the model sometimes repeats phrases or loops. The notebook uses temperature=0.8 and top_k=40 for the cleaner samples.
- **No evaluation beyond perplexity.** There's no BLEU, ROUGE, or human eval. Perplexity of ~11.4 is reasonable for a 30M model on this dataset, but it doesn't tell you much about coherence.

---

## 📁 Project Structure

```
├── oldermodel.ipynb          # ✅ Original model — fully trained & working
├── updated-model.ipynb       # 🚧 Upgraded architecture — untrained, WIP
├── readme.md                 # This file
└── .gitignore                # Git configuration
```

> **Note:** Files such as `model.py`, `train.py`, `train.bin`, `validation.bin`, and `best_model_params.pt` are for local development only and are not pushed to GitHub. The notebooks are self-contained.

---

## 🚀 How to Run

**Requirements:** Python 3.11+, Apple Silicon Mac (MPS backend), ~4GB free memory.

```bash
# Clone and set up
git clone https://github.com/Rhythem2005/NOVA-SLM.git
cd NOVA-SLM
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio datasets tiktoken matplotlib numpy

# Run the stable model
jupyter notebook oldermodel.ipynb

# Explore the updated architecture (untrained)
jupyter notebook updated-model.ipynb
```

Run cells top-to-bottom. The tokenization step (`train.bin`/`validation.bin`) only runs once — subsequent runs skip it. The training cell takes a while depending on your hardware.

---

## 🧠 What I Learned

This project was built to understand the GPT pipeline end-to-end:

- How BPE tokenization works and why we append `<|endoftext|>` as a story boundary signal
- Why pre-LN is more stable than post-LN for training
- What weight tying actually saves and why it helps
- How gradient accumulation simulates larger batch sizes on limited hardware
- Why bfloat16 doesn't need a GradScaler but float16 does
- How cosine LR decay with warmup prevents early training instability
- The difference between a model that's *memorized patterns* vs. one that *understands language* — and how far 30M parameters and 1,250 optimizer steps gets you (answer: further than you'd expect, but not far enough)
- Modern architectural improvements (RMSNorm, RoPE, GQA, SwiGLU) and how they compare to the classic GPT-2 approach


<p align="center">
  <i>Built with curiosity and PyTorch 🔥</i>
</p>
