"""
Attention implementation for SalesGPT.

This module provides efficient implementation of attention mechanisms with
graceful fallbacks when optimized libraries are not available.
"""
import math
import importlib.util
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for optional dependencies
USE_FLASH_ATTENTION = False
USE_XFORMERS = False

# Try to import flash_attn, but don't fail if not available
try:
    if importlib.util.find_spec("flash_attn") is not None:
        USE_FLASH_ATTENTION = True
        import flash_attn
except (ImportError, ModuleNotFoundError):
    print("FlashAttention not available, using standard attention mechanism")
    USE_FLASH_ATTENTION = False

# Try to import xformers, but don't fail if not available
try:
    if importlib.util.find_spec("xformers") is not None:
        USE_XFORMERS = True
        import xformers.ops
except (ImportError, ModuleNotFoundError):
    print("xformers not available, using standard attention mechanism")
    USE_XFORMERS = False


class FlashAttention(nn.Module):
    """
    Attention implementation with fallbacks.
    
    This implementation tries to use optimized attention mechanisms when available
    (FlashAttention2, xformers), falling back to standard PyTorch attention otherwise.
    
    Attributes:
        dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout probability
        causal (bool): Whether to use causal attention masking
        scale (float): Scaling factor for attention scores
        qkv_proj (nn.Linear): Linear projection for query, key, value
        out_proj (nn.Linear): Linear projection for output
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = True,
        qkv_bias: bool = True,
        xformers_fallback: bool = True,
    ):
        """
        Initialize attention module.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head (computed if None)
            dropout: Dropout probability
            causal: Whether to use causal attention masking
            qkv_bias: Whether to use bias for query, key, value projections
            xformers_fallback: Whether to try xformers before standard attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dropout = dropout
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.xformers_fallback = xformers_fallback and USE_XFORMERS
        
        # Query, key, value projections
        self.qkv_proj = nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=True)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if qkv_bias:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
        # Log which attention implementation will be used
        if USE_FLASH_ATTENTION:
            print("Using FlashAttention for efficient attention")
        elif self.xformers_fallback:
            print("Using xformers for efficient attention")
        else:
            print("Using standard PyTorch attention implementation")
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, values
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # [batch, 3, heads, seq_len, head_dim]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Determine which attention implementation to use
        if USE_FLASH_ATTENTION:
            try:
                from flash_attn import flash_attn_func
                
                # Reshape for flash_attn input requirements
                q = q.transpose(1, 2)  # [batch, seq_len, heads, head_dim]
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                
                # Apply softmax scaling
                q = q * self.scale
                
                # Call FlashAttention
                output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=self.causal,
                    softmax_scale=None,  # Already applied
                )
                
                # Reshape output
                output = output.reshape(batch_size, seq_len, -1)
                
            except Exception as e:
                print(f"FlashAttention failed: {e}. Falling back to alternative.")
                if self.xformers_fallback:
                    output = self._xformers_attention(q, k, v, attention_mask)
                else:
                    output = self._standard_attention(q, k, v, attention_mask)
        elif self.xformers_fallback:
            try:
                output = self._xformers_attention(q, k, v, attention_mask)
            except Exception as e:
                print(f"xformers attention failed: {e}. Falling back to standard attention.")
                output = self._standard_attention(q, k, v, attention_mask)
        else:
            output = self._standard_attention(q, k, v, attention_mask)
        
        # Project output
        output = self.out_proj(output)
        
        return output
    
    def _xformers_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention using xFormers library as fallback.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        if not USE_XFORMERS:
            return self._standard_attention(q, k, v, attention_mask)
            
        try:
            import xformers.ops as xops
            
            # Reshape for xFormers
            batch_size, num_heads, seq_len, head_dim = q.shape
            q = q.reshape(batch_size * num_heads, seq_len, head_dim)
            k = k.reshape(batch_size * num_heads, seq_len, head_dim)
            v = v.reshape(batch_size * num_heads, seq_len, head_dim)
            
            # Create attention bias for causal masking
            attn_bias = None
            if self.causal:
                attn_bias = xops.LowerTriangularMask()
            
            # Apply scaled attention
            q = q * self.scale
            
            # Compute attention
            output = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.dropout if self.training else 0.0,
            )
            
            # Reshape output
            output = output.reshape(batch_size, seq_len, -1)
            
            return output
        
        except Exception:
            return self._standard_attention(q, k, v, attention_mask)
    
    def _standard_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention using standard PyTorch operations.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-1, -2))  # [batch, heads, seq_len, seq_len]
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn + attention_mask
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to values
        output = torch.matmul(attn, v)  # [batch, heads, seq_len, head_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return output


class MHA(nn.Module):
    """
    Multi-head attention wrapper around attention implementation.
    
    This is a convenience wrapper to handle different attention implementations
    uniformly in the architecture.
    
    Attributes:
        attention (nn.Module): Underlying attention implementation
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        use_flash_attn: bool = True,
        qkv_bias: bool = True,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal attention masking
            use_flash_attn: Whether to try using FlashAttention
            qkv_bias: Whether to use bias for query, key, value projections
        """
        super().__init__()
        
        if use_flash_attn:
            self.attention = FlashAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                causal=causal,
                qkv_bias=qkv_bias,
                xformers_fallback=True,
            )
        else:
            # Fallback to standard multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=qkv_bias,
                batch_first=True,
            )
            self.causal = causal
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        if isinstance(self.attention, FlashAttention):
            return self.attention(x, attention_mask)
        else:
            # Standard multi-head attention
            if self.causal:
                # Create causal mask for standard attention
                seq_len = x.size(1)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                if attention_mask is not None:
                    attention_mask = attention_mask.logical_or(causal_mask)
                else:
                    attention_mask = causal_mask
            
            # Convert mask format if needed
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            
            output, _ = self.attention(x, x, x, attn_mask=attention_mask)
            return output