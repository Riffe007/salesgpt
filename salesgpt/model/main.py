"""
Main script for training and using the SalesGPT model.

This script provides examples for how to:
1. Initialize a SalesGPT model
2. Train it using PPO
3. Use it for inference
"""
import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from salesgpt.model.model import (
    SalesGPT,
    create_salesgpt_model,
    SALESGPT_CONFIG_TINY, 
    SALESGPT_CONFIG_SMALL
)
from salesgpt.training.ppo import PPOTrainer, create_sales_environment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_model(model_size: str = "tiny", device: str = "cuda"):
    """
    Create a SalesGPT model.
    
    Args:
        model_size: Size of model ("tiny", "small", "base", "large")
        device: Device to place model on
        
    Returns:
        SalesGPT model
    """
    logger.info(f"Creating {model_size} model...")
    
    # Choose config based on size
    if model_size == "tiny":
        config = SALESGPT_CONFIG_TINY
    elif model_size == "small":
        config = SALESGPT_CONFIG_SMALL
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    # Add RL-specific parameters to config
    config["use_value_head"] = True
    
    # Create model
    model = create_salesgpt_model(config)
    model = model.to(device)
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def train_with_ppo(
    model: SalesGPT,
    steps_per_epoch: int = 1000,
    epochs: int = 10,
    device: str = "cuda",
):
    """
    Train a model using PPO.
    
    Args:
        model: SalesGPT model to train
        steps_per_epoch: Steps per training epoch
        epochs: Number of training epochs
        device: Device to train on
    """
    logger.info("Starting PPO training...")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Create PPO trainer
    trainer = PPOTrainer(
        policy_model=model,
        optimizer=optimizer,
        config={
            "device": device,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "mini_batch_size": 16,
        },
    )
    
    # Run training
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        trainer.train(
            env_func=create_sales_environment,
            num_steps=steps_per_epoch,
            num_envs=1,
            total_timesteps=steps_per_epoch,
            log_interval=1,
        )


def demo_inference(model: SalesGPT, device: str = "cuda"):
    """
    Run a simple inference demo with the model.
    
    Args:
        model: SalesGPT model for inference
        device: Device to run inference on
    """
    logger.info("Running inference demo...")
    
    # Set model to eval mode
    model.eval()
    
    # This is just a placeholder - in a real implementation, you would:
    # 1. Use a proper tokenizer to encode the input
    # 2. Generate a response with the model
    # 3. Decode the response
    
    # Dummy input (would normally be tokenized text)
    input_tensor = torch.randint(0, model.vocab_size, (1, 10), device=device)
    
    # Generate output
    with torch.no_grad():
        output = model.generate(
            input_ids=input_tensor,
            max_new_tokens=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
    
    logger.info(f"Generated sequence of length {output.size(1)}")
    
    # In a real implementation, you would decode this output to text


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Train or run inference with SalesGPT")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "demo"],
                        help="Mode to run in: 'train' or 'demo'")
    parser.add_argument("--model-size", type=str, default="tiny", choices=["tiny", "small"],
                        help="Size of model to create")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training/inference")
    args = parser.parse_args()
    
    # Create model
    model = create_model(model_size=args.model_size, device=args.device)
    
    # Run in specified mode
    if args.mode == "train":
        train_with_ppo(model, device=args.device)
    elif args.mode == "demo":
        demo_inference(model, device=args.device)


if __name__ == "__main__":
    main()