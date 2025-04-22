# SalesGPT

A small language model specialized for software sales with emotional intelligence.
1. Clone this repository
2. Run the setup script:
3. Activate the virtual environment:
5. Run the application:
- Specialized for software sales conversations
- Incorporates emotional intelligence
- Built on PyTorch with advanced transformer architecture



Core Architecture Components - We've implemented the essential building blocks:

FlashAttention2 implementation in salesgpt/model/attention.py
Transformer Block in salesgpt/model/transformer.py
Complete Model Architecture in salesgpt/model/model.py
PPO Training System in salesgpt/training/ppo.py
Main Script in salesgpt/main.py for training and inference


Directory Structure - We've created the proper directory structure with empty __init__.py files:
```plaintext
salesgpt/
├── salesgpt/
│   ├── core/
│   ├── model/
│   ├── api/
│   ├── training/
│   └── utils/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
│   ├── unit/
│   └── integration/
├── requirements.txt
├── run.py
└── README.md
```
Next Steps
Here's what you'll want to do next:

Install Dependencies:
```bash
pip install torch numpy tqdm transformers sentence-transformers fastapi uvicorn pydantic
```
- Review the Code: Take some time to understand the implementations provided
Implement Additional Components:

- Tokenizer implementation for processing text
- Data loading and preprocessing for training
- The RAG system for retrieving information about your team


Run Training Tests: Start with tiny model size to ensure everything works
API Integration: Connect the model to the FastAPI endpoints we've set up

This implementation gives you a strong foundation with all the core components necessary for a transformer-based language model with PPO training. The model architecture includes the attention mechanisms, transformer blocks, and the overall decoder structure you need.


A small language model (100-150M parameters) specialized for software sales with emotional intelligence for negotiation.

## Project Overview

SalesGPT is an advanced small language model that:
- Specializes in software development consultative sales
- Incorporates emotional intelligence for negotiation
- Implements a transformer architecture from scratch
- Can be trained using reinforcement learning (PPO)
- Runs entirely locally on 8GB VRAM hardware

## Features

- **FlashAttention2**: Fast and memory-efficient attention mechanism
- **Memory Optimization**: Gradient checkpointing for training on consumer GPUs
- **PPO Implementation**: Reinforcement learning for optimizing sales conversations
- **Sales Environment**: Simulated environment for RL training
- **Configurable Model Sizes**: Multiple model configurations for different resource constraints

## Directory Structure
```plaintext
salesgpt/
├── salesgpt/
│   ├── core/            # Core infrastructure components
│   ├── model/           # Model architecture and components
│   │   ├── attention.py # Attention mechanisms (FlashAttention2)
│   │   ├── transformer.py # Transformer block implementation
│   │   └── model.py     # Complete model architecture
│   ├── training/        # Training components
│   │   └── ppo.py       # PPO implementation
│   ├── api/             # FastAPI application
│   │   └── main.py      # API endpoints
│   └── utils/           # Utility functions
├── data/                # Data storage
│   ├── raw/             # Raw data
│   └── processed/       # Processed data
├── tests/               # Test files
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── requirements.txt     # Project dependencies
├── run.py               # Script to run FastAPI server
└── main.py              # Main entry point for training and inference
```
## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/salesgpt.git
   cd salesgpt

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On Unix/MacOS
```
Install dependencies:
```bash
pip install -r requirements.txt
```

- Usage
- Training
- To train the model using PPO:
```bash
python main.py --mode train --model-size tiny
```
Options:

- mode: Mode to run in ('train' or 'demo')
- model-size: Size of model to create ('tiny' or 'small')
- device: Device to use for training/inference (default: 'cuda' if available)

- Inference Demo
- To run a simple inference demo:
```bash
python main.py --mode demo --model-size tiny
```
- API Server
- To start the FastAPI server:
```bash
python run.py
```
- The API will be available at http://localhost:8000
- Model Architecture
- SalesGPT is a decoder-only transformer model with:

- Token and positional embeddings
- Multiple transformer layers with FlashAttention2
- Layer normalization
- Optional value head for reinforcement learning

Available model sizes:
- SizeParametersLayersHeadsDimMax Seq - - - - - LenTiny~4M22128512Small~40M663841024Base~120M12127681024Large~350M241610242048
# Reinforcement Learning with PPO
The model can be trained using Proximal Policy Optimization (PPO) to optimize for effective sales conversations. The implementation includes:

- Experience buffer for collecting rollouts
- Generalized Advantage Estimation (GAE)
- Value function for estimating returns
- Sales environment simulation

# Future Work

 - Add tokenizer implementation
 - Implement RAG for retrieving information about your team's capabilities
 - Add support for fine-tuning on real sales conversations
 - Improve sales environment simulation with more realistic customer behavior
 - Add emotional intelligence components for response modulation

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
This project was inspired by recent advancements in language models and their application to specialized domains like sales.