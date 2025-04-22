"""
Transformer block implementation for SalesGPT.

This module provides the core transformer architecture components including
the transformer block, feed-forward network, and related utilities.
"""
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from salesgpt.model.attention import FlashAttention, MHA


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    This implements a standard MLP with GELU activation and optional dropout.
    
    Attributes:
        w1 (nn.Linear): First linear layer
        w2 (nn.Linear): Second linear layer
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Callable = F.gelu,
    ):
        """
        Initialize feed-forward network.
        
        Args:
            dim: Input and output dimension
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        return self.dropout(self.w2(self.activation(self.w1(x))))


class TransformerBlock(nn.Module):
    """
    Advanced transformer block implementation with memory optimization.
    
    This implementation includes:
    - FlashAttention2 for efficient attention
    - Memory-efficient feed-forward network
    - Gradient checkpointing support
    - Mixed precision compatibility
    
    Attributes:
        norm1 (nn.LayerNorm): First layer normalization
        norm2 (nn.LayerNorm): Second layer normalization (optional)
        attention (nn.Module): Attention mechanism
        mlp (FeedForward): Feed-forward network
        dropout (nn.Dropout): Dropout layer
        post_attention_layernorm (bool): Whether to use two layer norms
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layernorm_epsilon: float = 1e-5,
        activation: Callable = F.gelu,
        post_attention_layernorm: bool = False,
        attention_type: str = "flash",
    ):
        """
        Initialize the transformer block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Add bias to query, key, value projections
            dropout: Dropout rate
            attention_dropout: Attention-specific dropout rate
            layernorm_epsilon: Layer norm epsilon
            activation: Activation function
            post_attention_layernorm: Whether to use two layer norms
            attention_type: Type of attention mechanism to use
        """
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=layernorm_epsilon)
        self.norm2 = nn.LayerNorm(dim, eps=layernorm_epsilon) if post_attention_layernorm else None
        
        # Attention mechanism based on specified type
        if attention_type == "flash":
            self.attention = FlashAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout=attention_dropout,
                causal=True
            )
        elif attention_type == "mha":
            self.attention = MHA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout=attention_dropout,
                causal=True,
                use_flash_attn=False
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            activation=activation,
            dropout=dropout
        )
        
        # Other attributes
        self.dropout = nn.Dropout(dropout)
        self.post_attention_layernorm = post_attention_layernorm
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # First residual connection with attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Second residual connection with MLP
        residual = x
        if self.post_attention_layernorm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
    
    def forward_with_checkpointing(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing for memory efficiency.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Use checkpointing to save memory during training
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # First residual with checkpointing
        residual = x
        x = self.norm1(x)
        x = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.attention),
            x, attention_mask
        )
        x = self.dropout(x)
        x = residual + x
        
        # Second residual with checkpointing
        residual = x
        if self.post_attention_layernorm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.mlp),
            x
        )
        x = self.dropout(x)
        x = residual + x
        
        return x


class TransformerStack(nn.Module):
    """
    Stack of Transformer blocks.
    
    Attributes:
        layers (nn.ModuleList): List of transformer blocks
        norm (nn.LayerNorm): Final layer normalization
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layernorm_epsilon: float = 1e-5,
        activation: Callable = F.gelu,
        post_attention_layernorm: bool = False,
        attention_type: str = "flash",
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize the transformer stack.
        
        Args:
            dim: Hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Add bias to query, key, value projections
            dropout: Dropout rate
            attention_dropout: Attention-specific dropout rate
            layernorm_epsilon: Layer norm epsilon
            activation: Activation function
            post_attention_layernorm: Whether to use two layer norms
            attention_type: Type of attention mechanism to use
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        
        # Create transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layernorm_epsilon=layernorm_epsilon,
                activation=activation,
                post_attention_layernorm=post_attention_layernorm,
                attention_type=attention_type,
            )
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(dim, eps=layernorm_epsilon)
        
        # Save configuration
        self.gradient_checkpointing = gradient_checkpointing
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer stack.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Pass through each transformer block
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = layer.forward_with_checkpointing(x, attention_mask)
            else:
                x = layer(x, attention_mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x