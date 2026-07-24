# Nova — A ~30M Parameter Language Model

Nova is a GPT-style language model trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. No pretrained weights, no Hugging Face `Trainer`, no shortcuts — just PyTorch, a tokenizer, and an Apple Silicon GPU.

The goal was to understand the full pipeline: data → tokens → batches → transformer → loss → generation.

> **⚠️ Status note (in progress):** The codebase has just been upgraded to a modernized architecture (RoPE, RMSNorm, GQA, SwiGLU, QK-Norm — details below). The results, sample generations, and checkpoint shown in this README are still from the **previous architecture** (LayerNorm + learned positional embeddings + standard MHA + GELU MLP). The new architecture has not been trained/verified yet — that's next. A PR with the retrained results will follow once verified. Until then, treat the "Results" section below as belonging to the old model, not the code currently in `main.ipynb`.

---

## What's in the Notebook

Everything lives in [`main.ipynb`](main.ipynb). The notebook walks through:

1. **Dataset loading** — TinyStories from Hugging Face (~2.1M train stories, ~22K validation)
2. **Tokenization** — GPT-2 BPE via `tiktoken`, serialized to binary files (`train.bin`, `validation.bin`) as uint16 memmap arrays
3. **Model definition** — A decoder-only transformer written from scratch
4. **Training** — AdamW with linear warmup + cosine LR decay, gradient accumulation, mixed precision (bfloat16), gradient clipping
5. **Evaluation** — Loss estimation, perplexity, and multi-prompt text generation

---

## Architecture (current code)

| Component | Value |
|---|---|
| Type | Decoder-only transformer |
| Layers | 8 (default config) |
| Query heads | 6 |
| KV heads | 2 (Grouped Query Attention, 3:1 sharing) |
| Embedding dim | 384 (head dim = 64) |
| Context window | 256 tokens (default config) |
| Vocab size | 50,304 (GPT-2 BPE, padded to multiple of 64) |
| Dropout | 0.05 |
| Position encoding | RoPE (rotary), no learned position table |
| Normalization | RMSNorm (pre-norm) + QK-Norm on Q/K before attention |
| Feed-forward | SwiGLU (~8/3× hidden dim) instead of GELU MLP |
| Attention logits | Soft-capped (tanh, cap=50) |
| Output logits | Soft-capped (tanh, cap=30) |
| Generation | KV-cached — O(1) per new token instead of full recompute |
| Weight tying | Yes — token embeddings shared with the output head |

This is a from-scratch reimplementation of ideas from LLaMA/Mistral/Gemma 2, applied to a small (~30M) model. Exact parameter count depends on the config used for a given training run (e.g. `n_layer`, `block_size`, `bias` are all overridable).

> **Previous architecture** (what actually produced the results below): standard LayerNorm, learned positional embeddings (`wpe`), full multi-head attention, GELU MLP (4× hidden dim), PyTorch Flash Attention. No KV cache in `generate()`.

---

## Training Setup (from the previous run — see status note)

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
| Config used | `n_layer=6, n_head=6, n_embd=384, block_size=128, vocab_size=50257, bias=True` |

---

## Results (from the previous architecture's checkpoint)

**Final losses (from the best checkpoint):**

| Split | Loss | Perplexity |
|---|---|---|
| Train | 2.4227 | — |
| Val | 2.4321 | ~11.4 |

The train/val gap is tiny (~0.01), which means the model is **not overfitting** — if anything, it's significantly **undertrained** (see Limitations below).

### Sample Generations

These are actual outputs from the notebook, generated from the best checkpoint (previous architecture):

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

**Note:** `best_model_params.pt` in this repo is a checkpoint from the previous architecture. It is **not compatible** with the current `GPT`/`GPTConfig` in `main.ipynb` (different module names — `q_proj`/`k_proj`/`gate_proj` vs. the old `c_attn`/`c_fc`, no `wpe`, etc.). Loading it into the new architecture will raise a state-dict key mismatch. A fresh checkpoint is needed once the new architecture is trained.

---

## Limitations

This is a learning project, not a production model. Here's what's honest:

- **New architecture unverified.** RoPE/GQA/RMSNorm/SwiGLU are in the code but haven't been trained or benchmarked yet on this dataset/hardware.
- **Undertrained (previous run).** 5,000 micro-steps is ~1,250 actual optimizer updates. The loss was almost certainly still decreasing when training stopped.
- **Tiny context.** 128–256 tokens is short. Stories that exceed this window lose all prior context.
- **Semantic drift.** The model generates plausible-sounding sentences that often don't follow logically from each other.
- **Repetition.** With default sampling, the model sometimes repeats phrases or loops.
- **No evaluation beyond perplexity.** There's no BLEU, ROUGE, or human eval.
- **Loss curve not captured (previous run).** The plotting cell errors because `train_loss_list` wasn't in memory in that session.

---

## Project Structure

```
├── main.ipynb              # Everything: data, model, training, eval, generation
├── train.bin                # Tokenized training data (memmap, uint16)
├── validation.bin           # Tokenized validation data (memmap, uint16)
├── best_model_params.pt     # Checkpoint from the PREVIOUS architecture (incompatible with current code)
└── readme.md                 # This file
```

> **Note:** Files such as `model.py`, and `train.py` are for personal use / local development and are git-ignored.

---

## How to Run

**Requirements:** Python 3.11+, Apple Silicon Mac (MPS backend), ~4GB free memory.

```bash
# Clone and set up
git clone https://github.com/Rhythem2005/NOVA-SLM.git
cd NOVA-SLM
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio datasets tiktoken matplotlib

# Open the notebook
jupyter notebook main.ipynb
```

Run cells top-to-bottom. The tokenization step (`train.bin`/`validation.bin`) only runs once — subsequent runs skip it. Because the checkpoint on disk belongs to the old architecture, delete/ignore `best_model_params.pt` and retrain from scratch to get a checkpoint compatible with the current code.

---
