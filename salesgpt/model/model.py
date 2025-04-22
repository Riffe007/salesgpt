"""
SalesGPT model architecture.

This module defines the core model architecture for SalesGPT, implementing
a decoder-only transformer model specialized for sales conversations.
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from salesgpt.model.transformer import TransformerBlock, TransformerStack


class SalesGPT(nn.Module):
    """
    SalesGPT core model architecture.
    
    This is a decoder-only transformer model with:
    - Token and positional embeddings
    - Transformer layers with efficient attention
    - Output projection for next token prediction
    - Optional value head for reinforcement learning
    
    Attributes:
        token_embedding (nn.Embedding): Token embedding layer
        position_embedding (nn.Embedding): Position embedding layer
        transformer (TransformerStack): Stack of transformer blocks
        ln_f (nn.LayerNorm): Final layer normalization
        lm_head (nn.Linear): Language model head for next token prediction
        value_head (nn.Linear): Optional value head for RL training
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        use_value_head: bool = False,
        pad_token_id: int = 0,
        attention_type: str = "flash",
        gradient_checkpointing: bool = False,
        layernorm_epsilon: float = 1e-5,
    ):
        """
        Initialize SalesGPT model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            attention_dropout: Dropout rate specifically for attention
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            use_value_head: Whether to include a value head for RL
            pad_token_id: Token ID for padding
            attention_type: Type of attention mechanism ("flash" or "mha")
            gradient_checkpointing: Whether to use gradient checkpointing
            layernorm_epsilon: Epsilon for layer normalization
        """
        super().__init__()
        
        # Save configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.use_value_head = use_value_head
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.transformer = TransformerStack(
            dim=d_model,
            depth=n_layer,
            num_heads=n_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layernorm_epsilon=layernorm_epsilon,
            attention_type=attention_type,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model, eps=layernorm_epsilon)
        
        # Output heads
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Optional value head for RL
        self.value_head = nn.Linear(d_model, 1) if use_value_head else None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights between token embedding and LM head
        self.lm_head.weight = self.token_embedding.weight
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize model weights.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            # Special scaled init for the output layer
            if module is self.lm_head:
                module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        return_value: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
                (1 for tokens to attend to, 0 for masked tokens)
            position_ids: Position IDs [batch_size, seq_len]
            return_dict: Whether to return a dictionary of outputs
            return_value: Whether to return the value head output
            
        Returns:
            If return_dict is True:
                Dictionary with "logits" and optionally "values"
            If return_dict is False and return_value is False:
                Token logits [batch_size, seq_len, vocab_size]
            If return_dict is False and return_value is True:
                (logits, values) tuple
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        # Convert attention mask to the format expected by transformer
        # [batch_size, 1, seq_len, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        
        # Get token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Pass through transformer
        hidden_states = self.transformer(hidden_states, attention_mask=extended_attention_mask)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Get values if needed
        values = None
        if return_value and self.value_head is not None:
            values = self.value_head(hidden_states)
        
        # Return outputs based on parameters
        if return_dict:
            outputs = {"logits": logits}
            if values is not None:
                outputs["values"] = values
            return outputs
        elif return_value and values is not None:
            return logits, values
        else:
            return logits
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text from a prompt.
        
        Args:
            input_ids: Prompt token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider for sampling
            top_p: Probability threshold for nucleus sampling
            do_sample: Whether to sample or greedy decode
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end-of-sequence
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        
        # Set model to eval mode
        self.eval()
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        # Keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Clone input_ids to avoid modifying the original
        output_ids = input_ids.clone()
        
        # Generate until max_new_tokens or all sequences are finished
        for _ in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                logits = self(
                    input_ids=output_ids,
                    return_dict=False,
                    return_value=False,
                )
            
            # Only use the logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                # Get top-k logits and their indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Create a mask for the top-k logits
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                
                # Calculate cumulative probabilities
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a scatter mask for the indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                
                # Set the logits for the indices to remove to -inf
                next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
            
            # Sample or greedy decode
            if do_sample:
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Check if any sequences are finished
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id)
            
            # Add the new tokens to the output
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Stop if all sequences are finished
            if torch.all(unfinished_sequences == 0):
                break
        
        return output_ids


class SalesGPTConfig:
    """Configuration for SalesGPT model."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        use_value_head: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        attention_type: str = "flash",
        gradient_checkpointing: bool = False,
        layernorm_epsilon: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
    ):
        """
        Initialize SalesGPT configuration.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            attention_dropout: Dropout rate specifically for attention
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            use_value_head: Whether to include a value head for RL
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end-of-sequence
            attention_type: Type of attention mechanism ("flash" or "mha")
            gradient_checkpointing: Whether to use gradient checkpointing
            layernorm_epsilon: Epsilon for layer normalization
            use_cache: Whether to use kv-cache during generation
            tie_word_embeddings: Whether to tie input and output embeddings
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mlp_ratio = mlp_ratio
        self.use_value_head = use_value_head
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.attention_type = attention_type
        self.gradient_checkpointing = gradient_checkpointing
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SalesGPTConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            SalesGPT configuration
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "mlp_ratio": self.mlp_ratio,
            "use_value_head": self.use_value_head,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "attention_type": self.attention_type,
            "gradient_checkpointing": self.gradient_checkpointing,
            "layernorm_epsilon": self.layernorm_epsilon,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def create_salesgpt_model(config: Union[Dict[str, Any], SalesGPTConfig]) -> SalesGPT:
    """
    Create a SalesGPT model from configuration.
    
    Args:
        config: Model configuration dictionary or object
        
    Returns:
        SalesGPT model
    """
    # Convert dictionary to config object if needed
    if isinstance(config, dict):
        config = SalesGPTConfig.from_dict(config)
    
    # Create model
    model = SalesGPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layer=config.n_layer,
        n_head=config.n_head,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        mlp_ratio=config.mlp_ratio,
        use_value_head=config.use_value_head,
        pad_token_id=config.pad_token_id,
        attention_type=config.attention_type,
        gradient_checkpointing=config.gradient_checkpointing,
        layernorm_epsilon=config.layernorm_epsilon,
    )
    
    return model


# Example configurations for different model sizes
SALESGPT_CONFIG_TINY = {
    "vocab_size": 32000,
    "d_model": 128,
    "n_layer": 2,
    "n_head": 2,
    "max_seq_len": 512,
}

SALESGPT_CONFIG_SMALL = {
    "vocab_size": 32000,
    "d_model": 384,
    "n_layer": 6,
    "n_head": 6,
    "max_seq_len": 1024,
}

SALESGPT_CONFIG_BASE = {
    "vocab_size": 32000,
    "d_model": 768,
    "n_layer": 12,
    "n_head": 12,
    "max_seq_len": 1024,
}

SALESGPT_CONFIG_LARGE = {
    "vocab_size": 32000,
    "d_model": 1024,
    "n_layer": 24,
    "n_head": 16,
    "max_seq_len": 2048,
}