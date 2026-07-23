"""
Nova SLM — Training Script
===========================
Complete training pipeline for the optimized Nova SLM architecture.
Run this script directly, or copy the relevant cells into main.ipynb.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from contextlib import nullcontext
from tqdm.auto import tqdm

from model import GPT, GPTConfig


# ===========================================================================
# Configuration
# ===========================================================================

config = GPTConfig(
    vocab_size=50304,       # GPT-2 vocab (50257) padded to multiple of 64
    block_size=256,         # context window (doubled from 128)
    n_layer=8,              # deeper network (was 6)
    n_head=6,               # query heads
    n_kv_head=2,            # KV heads for GQA
    n_embd=384,             # embedding dimension
    dropout=0.05,           # lower dropout (model was undertrained)
    bias=False,             # bias-free design
)

learning_rate = 3e-4        # peak learning rate
min_lr = 1e-5               # minimum learning rate after cosine decay
max_iters = 5000            # total micro-steps
warmup_steps = 200          # linear warmup steps
eval_iters = 250            # evaluate every N micro-steps
eval_batches = 200          # number of batches for loss estimation
batch_size = 32             # micro-batch size
block_size = config.block_size
gradient_accumulation_steps = 4  # effective batch = 32 * 4 = 128
max_grad_norm = 1.0         # gradient clipping

device = "mps" if torch.backends.mps.is_available() else "cpu"
device_type = "mps" if device == "mps" else "cpu"
dtype = "bfloat16"
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != "cpu" else nullcontext()


# ===========================================================================
# Data Loading
# ===========================================================================

def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_batches)
            for k in range(eval_batches):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


# ===========================================================================
# Training
# ===========================================================================

def train():
    torch.manual_seed(42)

    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
    )

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler("mps", enabled=(dtype == 'float16' and device == 'mps'))

    best_val_loss = float("inf")
    best_model_params_path = "best_model_params.pt"
    train_loss_list, validation_loss_list = [], []

    model.train()
    print(f"\nStarting training: {max_iters} micro-steps, "
          f"effective batch size = {batch_size * gradient_accumulation_steps}")

    t0 = time.time()

    for epoch in tqdm(range(max_iters), desc="Training"):
        if epoch % eval_iters == 0 and epoch != 0:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"\nStep {epoch} ({elapsed:.1f}s): "
                  f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['val'])

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)
                print(f"  → Saved best model to {best_model_params_path}")

        X, y = get_batch("train")
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    losses = estimate_loss(model)
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    return model, train_loss_list, validation_loss_list


if __name__ == "__main__":
    if os.path.exists("train.bin"):
        train()
