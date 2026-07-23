import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Model configuration with modern defaults for a ~31M parameter SLM."""
    block_size: int = 256           # context window (doubled from 128)
    vocab_size: int = 50304         # GPT-2 vocab (50257) padded to multiple of 64
    n_layer: int = 8                # deeper network (was 6)
    n_head: int = 6                 # query heads
    n_kv_head: int = 2              # key/value heads for GQA (3 Q heads per KV group)
    n_embd: int = 384               # embedding dimension
    dropout: float = 0.05           # lower dropout (model was undertrained, not overfitting)
    bias: bool = False              # bias-free design (LLaMA/Mistral style)
    norm_eps: float = 1e-6          # RMSNorm epsilon
    rope_theta: float = 10000.0     # RoPE base frequency
    qk_norm: bool = True            # apply RMSNorm to Q and K (Gemma 2 style)
    attn_logit_cap: float = 50.0    # soft-cap attention logits (0 = disabled)
    final_logit_cap: float = 30.0   # soft-cap output logits (0 = disabled)
    intermediate_size: int = 0      # SwiGLU hidden dim (auto-computed if 0)


# ---------------------------------------------------------------------------
# RMSNorm — faster and simpler than LayerNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    ~15% faster than LayerNorm: no mean subtraction, no bias.
    Used by LLaMA, Mistral, Gemma, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability during norm computation
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE) — Su et al., 2021
# ---------------------------------------------------------------------------

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex exponential frequencies for RoPE.

    Returns a tensor of shape (max_seq_len, head_dim // 2) of complex64,
    representing e^{i * pos * freq} for each position and frequency band.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequency bands: theta^{-2k/d} for k = 0, 1, ..., d/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Position indices
    t = torch.arange(max_seq_len, device=device).float()
    # Outer product: (seq_len, head_dim/2)
    freqs = torch.outer(t, freqs)
    # Complex exponentials for efficient rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{i*theta}
    return freqs_cis


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor x.

    Args:
        x: (B, n_heads, T, head_dim) — query or key tensor
        freqs_cis: (T, head_dim // 2) — precomputed complex frequencies
    """
    B, H, T, D = x.shape
    # Reshape to pairs: (B, H, T, D/2, 2) then view as complex
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # Broadcast freqs: (1, 1, T, D/2)
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    # Apply rotation via complex multiplication and convert back to real
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, T, D)
    return x_rotated.to(x.dtype)


# ---------------------------------------------------------------------------
# KV Cache — for efficient autoregressive generation
# ---------------------------------------------------------------------------

class KVCache:
    """Simple KV cache for autoregressive generation.

    Stores past key and value tensors to avoid redundant recomputation.
    Reduces generation from O(n²) to O(n) per new token.
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K, V to cache and return full K, V."""
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.size(2)

    def reset(self):
        self.k = None
        self.v = None


# ---------------------------------------------------------------------------
# Grouped Query Attention with RoPE and QK-Norm
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA), RoPE, and QK-Norm.

    Key improvements over standard MHA:
    - GQA: fewer KV heads → smaller KV cache, fewer parameters, faster inference
    - RoPE: relative position encoding in Q/K dot products, no learned pos embeddings
    - QK-Norm: RMSNorm on Q and K to prevent attention logit explosion
    - Logit soft-capping: prevents extreme attention weights
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head  # repetition factor for GQA
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.attn_logit_cap = config.attn_logit_cap

        # Separate Q, K, V projections (cleaner than fused for GQA)
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # QK-Norm (Gemma 2 style) — normalizes Q and K before dot product
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Check for Flash Attention support
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q head count for GQA.

        (B, n_kv_head, T, D) → (B, n_head, T, D)
        """
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x.unsqueeze(2).expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK-Norm: normalize before RoPE (Gemma 2 approach)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # KV Cache: append and retrieve full K, V
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand KV heads to match Q heads (GQA)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Attention computation
        if self.use_flash and kv_cache is None and self.attn_logit_cap == 0:
            # Use Flash Attention when possible (no KV cache, no logit capping)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention for KV cache or logit capping scenarios
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            # Logit soft-capping (Gemma 2): tanh(att / cap) * cap
            if self.attn_logit_cap > 0:
                att = self.attn_logit_cap * torch.tanh(att / self.attn_logit_cap)

            # Causal mask
            T_q, T_k = q.size(2), k.size(2)
            # Build causal mask: each query position can attend to all key positions up to its own
            causal_mask = torch.triu(
                torch.full((T_q, T_k), float('-inf'), device=q.device, dtype=q.dtype),
                diagonal=T_k - T_q + 1,
            )
            att = att + causal_mask.unsqueeze(0).unsqueeze(0)

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape and output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network — PaLM, LLaMA style
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    SwiGLU(x) = SiLU(W_gate @ x) * (W_up @ x)
    output = W_down @ SwiGLU(x)

    Consistently outperforms GELU/ReLU in language modeling benchmarks.
    Hidden dimension is ~8/3 * n_embd (rounded to multiple of 64) to maintain
    similar parameter count to 4x GELU MLP while being more expressive.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Compute intermediate size: 8/3 * n_embd rounded to multiple of 64
        if config.intermediate_size > 0:
            hidden_dim = config.intermediate_size
        else:
            hidden_dim = int(8 * config.n_embd / 3)
            # Round up to nearest multiple of 64 for GPU efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise product of gated and ungated paths
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block — Pre-Norm Residual
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block with pre-norm residual connections.

    Architecture: x → RMSNorm → Attention → + residual
                  x → RMSNorm → SwiGLU FFN → + residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freqs_cis, kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model — Full Decoder-Only Transformer
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Modern decoder-only transformer language model.

    Combines:
    - Token embeddings with weight tying to lm_head
    - RoPE (no learned position embeddings)
    - N transformer blocks with GQA + SwiGLU
    - RMSNorm final layer norm
    - Optional logit soft-capping
    - KV-cached autoregressive generation
    """

    def __init__(self, config: Optional[GPTConfig] = None):
        super().__init__()
        if config is None:
            config = GPTConfig()
        self.config = config

        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies (not a learned parameter)
        self.register_buffer(
            "freqs_cis",
            precompute_rope_frequencies(head_dim, config.block_size * 2, config.rope_theta),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) for stable deep networks
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_tie = n_params - self.lm_head.weight.numel()
        print(f"Nova SLM initialized: {n_params:,} parameters ({n_params_no_tie:,} non-tied)")

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-3/LLaMA best practices."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: Token indices, shape (B, T)
            targets: Target token indices for loss computation, shape (B, T)
            kv_caches: List of KVCache objects (one per layer) for generation

        Returns:
            logits: (B, T, vocab_size) if targets given, else (B, 1, vocab_size)
            loss: scalar loss if targets given, else None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Determine position offset for RoPE (accounts for KV cache)
        start_pos = 0
        if kv_caches is not None and kv_caches[0].seq_len > 0:
            start_pos = kv_caches[0].seq_len

        # Token embeddings (no position embeddings — RoPE handles positions)
        x = self.transformer.drop(self.transformer.wte(idx))

        # Get RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        # Transformer blocks
        for i, block in enumerate(self.transformer.h):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, freqs_cis, cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)

            # Optional: logit soft-capping (Gemma 2)
            if self.config.final_logit_cap > 0:
                logits = self.config.final_logit_cap * torch.tanh(logits / self.config.final_logit_cap)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])

            if self.config.final_logit_cap > 0:
                logits = self.config.final_logit_cap * torch.tanh(logits / self.config.final_logit_cap)

            return logits, None

    def _create_kv_caches(self) -> list:
        """Create a fresh set of KV caches for generation."""
        return [KVCache() for _ in range(self.config.n_layer)]

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV cache.

        Args:
            idx: Conditioning sequence, shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Keep only top-k logits (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            repetition_penalty: Penalize repeated tokens (1.0 = disabled)
            eos_token_id: Stop generation when this token is produced

        Returns:
            Generated sequence including the conditioning prefix, shape (B, T + generated)
        """
        # Clamp temperature to avoid division by zero
        temperature = max(temperature, 1e-7)

        # Process the full prompt through the model to fill KV caches
        kv_caches = self._create_kv_caches()

        # Prefill: process the entire prompt at once
        prompt = idx
        if prompt.size(1) > self.config.block_size:
            prompt = prompt[:, -self.config.block_size:]

        logits, _ = self(prompt, kv_caches=kv_caches)

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get logits for the last position
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty: reduce probability of already-generated tokens
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    prev_tokens = idx[b].unique()
                    for token_id in prev_tokens:
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                v, _ = torch.topk(next_logits, k)
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                # Scatter back
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Early stopping on EOS
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

            # Check block_size limit for KV cache
            if kv_caches[0].seq_len >= self.config.block_size:
                break

            # Forward pass for the single new token (KV cache makes this O(1))
            logits, _ = self(idx_next, kv_caches=kv_caches)

        return idx

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        For non-embedding count (default), the token embedding parameters
        are subtracted since they are tied with lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
