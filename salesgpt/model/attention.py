# src/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def generate_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Creates a (seq_len x seq_len) causal mask with 0 on and below diagonal, -inf above.
    """
    # Lower triangular matrix of ones
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=dtype))
    # Convert 1→0 and 0→-inf
    return (1.0 - mask) * torch.finfo(dtype).min

class FlashAttention2(nn.Module):
    """
    Pure‑PyTorch FlashAttention‑2 style multi‑head attention.

    Features:
    - fused QKV projection
    - scaled‑dot‑prod with PyTorch 2.x kernel
    - causal masking
    - optional block‑wise chunking for long seqs
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        chunk_size: Optional[int] = None
    ):
        """
        :param embed_dim: dimensionality of input embeddings
        :param num_heads: how many attention heads
        :param dropout: dropout probability on attention weights
        :param causal: whether to apply causal mask (for autoregressive)
        :param chunk_size: if set, will process sequence in blocks of this size
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.causal = causal
        self.chunk_size = chunk_size

        # Single linear for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, seq_len, embed_dim)
        :returns: (batch, seq_len, embed_dim)
        """
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype

        # 1) project to QKV: shape (B, N, 3, num_heads, head_dim)
        qkv = self.qkv_proj(x) \
                 .view(B, N, 3, self.num_heads, self.head_dim) \
                 .permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, heads, N, head_dim)

        # 2) apply scaling
        q = q * self.scale

        # 3) determine mask or chunking
        attn_mask = None
        if self.causal:
            attn_mask = generate_causal_mask(N, device, dtype)  # (N, N)

        # If chunking, process in blocks
        if self.chunk_size and self.chunk_size < N:
            return self._chunked_attention(q, k, v, attn_mask)

        # 4) fused scaled-dot-prod with PyTorch
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # we've already applied mask
        )  # (B, heads, N, head_dim)

        # 5) recombine heads and project
        out = attn_out.transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(out)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Block‑wise attention to keep memory footprint ~O(B * heads * chunk^2)
        rather than full O(B * heads * N^2).
        """
        B, H, N, D = q.shape
        cs = self.chunk_size
        outputs = []

        # process each query block
        for start in range(0, N, cs):
            end = min(start + cs, N)
            q_block = q[..., start:end, :]                          # (B, H, cs, D)
            # compute attention against full K,V
            # adjust mask if causal
            block_mask = None
            if attn_mask is not None:
                block_mask = attn_mask[start:end, :]                # (cs, N)

            # scaled_dot_prod over this block
            out_block = F.scaled_dot_product_attention(
                q_block, k, v,
                attn_mask=block_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B, H, cs, D)
            outputs.append(out_block)

        # concatenate blocks: list of (B, H, cs, D) → (B, H, N, D)
        attn_out = torch.cat(outputs, dim=2)
        # reshape & project
        out = attn_out.transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(out)
