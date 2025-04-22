"""
PPO (Proximal Policy Optimization) implementation for SalesGPT.

This module implements the PPO algorithm for reinforcement learning with
language models, focusing on optimizing sales conversations by learning
from feedback and rewards.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class PPOBatch:
    """
    Batch of data for PPO training.
    
    Attributes:
        states: States or observations
        actions: Actions taken
        old_log_probs: Log probabilities of actions under old policy
        rewards: Rewards received
        returns: Computed returns (sum of discounted rewards)
        values: Value estimates
        advantages: Computed advantages
        masks: Masks for padding or episode boundaries
    """
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    masks: torch.Tensor


class ExperienceBuffer:
    """
    Buffer for storing experience trajectories for PPO training.
    
    This class manages a buffer of experiences (state, action, reward, etc.)
    collected during environment interaction, with utilities for computing
    returns and advantages.
    """
    
    def __init__(self, buffer_size: int = 2048, device: str = "cuda"):
        """
        Initialize experience buffer.
        
        Args:
            buffer_size: Maximum number of steps to store
            device: Device to store tensors on
        """
        self.device = device
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.ref_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        ref_log_prob: Optional[float] = None,
        done: bool = False,
        info: Optional[Dict] = None,
    ) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Environment state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action under current policy
            ref_log_prob: Log probability of action under reference policy (for KL)
            done: Whether this is a terminal state
            info: Additional information
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        if ref_log_prob is not None:
            self.ref_log_probs.append(ref_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute returns and advantages for the collected experiences.
        
        Uses Generalized Advantage Estimation (GAE) to compute advantages.
        
        Args:
            last_value: Value estimate for the final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        # Initialize arrays
        self.returns = np.zeros_like(rewards)
        self.advantages = np.zeros_like(rewards)
        
        # Compute GAE advantages and returns
        gae = 0
        for t in reversed(range(len(rewards))):
            # For episode boundaries, reset advantage
            if t == len(rewards) - 1 or dones[t]:
                next_value = last_value if t == len(rewards) - 1 else values[t + 1]
                next_non_terminal = 0.0 if dones[t] else 1.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # Compute TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # Compute GAE advantage
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            
            # Compute return (for critic loss)
            self.returns[t] = gae + values[t]
    
    def get_batch(self, batch_size: int = 64, normalize_advantages: bool = True) -> PPOBatch:
        """
        Get a batch of data for PPO training.
        
        Args:
            batch_size: Batch size
            normalize_advantages: Whether to normalize advantages
            
        Returns:
            Batch of training data
        """
        # Convert to tensors
        states = torch.tensor(self.states, device=self.device)
        actions = torch.tensor(self.actions, device=self.device)
        old_log_probs = torch.tensor(self.log_probs, device=self.device)
        returns = torch.tensor(self.returns, device=self.device, dtype=torch.float32)
        values = torch.tensor(self.values, device=self.device, dtype=torch.float32)
        advantages = torch.tensor(self.advantages, device=self.device, dtype=torch.float32)
        masks = torch.tensor(~np.array(self.dones), device=self.device, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32)
        
        # Normalize advantages if requested
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create batch
        return PPOBatch(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            returns=returns,
            values=values,
            advantages=advantages,
            masks=masks,
        )
    
    def __len__(self) -> int:
        """Get buffer length."""
        return len(self.states)


class PPOTrainer:
    """
    PPO trainer for language models.
    
    This implementation follows the PPO algorithm with adaptations
    for language model training, including:
    - Specialized reward models for sales conversations
    - Token-level rewards and advantages
    - Efficient KL divergence handling with reference model
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        value_optimizer: Optional[Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy_model: Policy model to train
            value_model: Value model (can be separate or shared with policy)
            ref_model: Reference model for KL divergence
            optimizer: Optimizer for policy model
            value_optimizer: Optimizer for value model (if separate)
            config: Configuration dictionary
        """
        self.policy_model = policy_model
        self.value_model = value_model if value_model is not None else policy_model
        self.ref_model = ref_model
        
        # Set up configuration with defaults
        self.config = {
            "lr": 5e-6,
            "eps": 1e-8,
            "value_lr": 5e-6,
            "clip_param": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "mini_batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "kl_target": 0.02,
            "kl_coef": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if config:
            self.config.update(config)
        
        # Create optimizers if not provided
        if optimizer is None:
            self.optimizer = Adam(
                self.policy_model.parameters(),
                lr=self.config["lr"],
                eps=self.config["eps"],
            )
        else:
            self.optimizer = optimizer
        
        # Create value optimizer if not provided and using separate value model
        if value_optimizer is None and value_model is not None and value_model != policy_model:
            self.value_optimizer = Adam(
                self.value_model.parameters(),
                lr=self.config.get("value_lr", self.config["lr"]),
                eps=self.config["eps"],
            )
        else:
            self.value_optimizer = value_optimizer
        
        # Initialize experience buffer
        self.buffer = ExperienceBuffer(
            buffer_size=self.config.get("buffer_size", 2048),
            device=self.config["device"],
        )
    
    def train_on_batch(self, batch: PPOBatch) -> Dict[str, float]:
        """
        Train on a batch of data.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Training metrics
        """
        # Move models to training mode
        self.policy_model.train()
        if self.value_model != self.policy_model:
            self.value_model.train()
        
        # Initialize metrics
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "loss": 0.0,
            "explained_variance": 0.0,
        }
        
        # Create dataset from batch
        dataset = TensorDataset(
            batch.states, batch.actions, batch.old_log_probs,
            batch.returns, batch.values, batch.advantages, batch.masks
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["mini_batch_size"],
            shuffle=True,
        )
        
        # Train for multiple epochs
        for _ in range(self.config["ppo_epochs"]):
            for states, actions, old_log_probs, returns, values, advantages, masks in dataloader:
                # Forward pass through policy model
                logits = self.policy_model(states)
                
                # Compute log probabilities and entropy
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Compute value estimates
                if self.value_model == self.policy_model:
                    value_preds = self.value_model(states, return_value=True)[1].squeeze(-1)
                else:
                    value_preds = self.value_model(states).squeeze(-1)
                
                # Compute KL divergence with reference model if available
                kl_divergence = 0.0
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_logits = self.ref_model(states)
                        ref_dist = torch.distributions.Categorical(logits=ref_logits)
                        kl_divergence = (
                            torch.distributions.kl_divergence(dist, ref_dist).mean().item()
                        )
                
                # Compute policy loss with clipping
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config["clip_param"],
                    1.0 + self.config["clip_param"],
                )
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages,
                ).mean()
                
                # Compute value loss with clipping
                value_loss = F.mse_loss(value_preds, returns)
                
                # Compute approximate KL
                approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean().item()
                
                # Compute clip fraction for monitoring
                clip_fraction = (
                    (ratio - 1.0).abs() > self.config["clip_param"]
                ).float().mean().item()
                
                # Compute total loss
                loss = (
                    policy_loss
                    + self.config["value_loss_coef"] * value_loss
                    - self.config["entropy_coef"] * entropy
                    + self.config["kl_coef"] * kl_divergence
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                if self.value_optimizer is not None:
                    self.value_optimizer.zero_grad()
                
                loss.backward()
                
                # Clip gradients
                if self.config["max_grad_norm"] > 0:
                    nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config["max_grad_norm"],
                    )
                    if self.value_model != self.policy_model and self.value_optimizer is not None:
                        nn.utils.clip_grad_norm_(
                            self.value_model.parameters(),
                            self.config["max_grad_norm"],
                        )
                
                # Optimize
                self.optimizer.step()
                if self.value_optimizer is not None:
                    self.value_optimizer.step()
                
                # Update metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl_divergence"] += kl_divergence
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_fraction
                metrics["loss"] += loss.item()
        
        # Average metrics over steps
        num_updates = len(dataloader) * self.config["ppo_epochs"]
        for k in metrics:
            metrics[k] /= num_updates
        
        # Compute explained variance
        with torch.no_grad():
            var_y = batch.returns.var().item()
            explained_var = 1 - (batch.returns - batch.values).var().item() / var_y
            metrics["explained_variance"] = explained_var
        
        return metrics
    
    def collect_rollouts(
        self,
        env_func: Callable,
        num_steps: int = 1000,
        num_envs: int = 1,
    ) -> None:
        """
        Collect rollouts from environments.
        
        Args:
            env_func: Function that returns an environment
            num_steps: Number of steps to collect per environment
            num_envs: Number of parallel environments
        """
        # Clear buffer
        self.buffer.clear()
        
        # Create environments
        envs = [env_func() for _ in range(num_envs)]
        
        # Initialize states
        states = [env.reset() for env in envs]
        
        # Collect experiences
        steps_collected = 0
        while steps_collected < num_steps:
            # Convert states to tensors
            state_batch = torch.stack([
                torch.tensor(state, device=self.config["device"])
                for state in states
            ])
            
            # Get actions and values from policy
            with torch.no_grad():
                # Get policy and value outputs
                logits = self.policy_model(state_batch)
                if self.value_model == self.policy_model:
                    _, values = self.policy_model(state_batch, return_value=True)
                    values = values.squeeze(-1)
                else:
                    values = self.value_model(state_batch).squeeze(-1)
                
                # Sample actions
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
                # Get log probs from reference model if available
                if self.ref_model is not None:
                    ref_logits = self.ref_model(state_batch)
                    ref_dist = torch.distributions.Categorical(logits=ref_logits)
                    ref_log_probs = ref_dist.log_prob(actions)
                else:
                    ref_log_probs = [None] * len(actions)
            
            # Execute actions in environments
            next_states, rewards, dones, infos = [], [], [], []
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, done, info = env.step(action.item())
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            
            # Store experiences
            for i in range(len(envs)):
                self.buffer.add(
                    state=states[i],
                    action=actions[i].item(),
                    reward=rewards[i],
                    value=values[i].item(),
                    log_prob=log_probs[i].item(),
                    ref_log_prob=ref_log_probs[i].item() if ref_log_probs[i] is not None else None,
                    done=dones[i],
                    info=infos[i],
                )
            
            # Update states
            states = [
                envs[i].reset() if dones[i] else next_states[i]
                for i in range(len(envs))
            ]
            
            steps_collected += len(envs)
        
        # Compute advantages
        with torch.no_grad():
            if len(next_states) > 0:
                last_states = torch.stack([
                    torch.tensor(state, device=self.config["device"])
                    for state in next_states
                ])
                
                if self.value_model == self.policy_model:
                    _, last_values = self.policy_model(last_states, return_value=True)
                    last_values = last_values.squeeze(-1)
                else:
                    last_values = self.value_model(last_states).squeeze(-1)
                
                for i, done in enumerate(dones):
                    if not done:
                        self.buffer.compute_returns_and_advantages(
                            last_value=last_values[i].item(),
                            gamma=self.config["gamma"],
                            gae_lambda=self.config["gae_lambda"],
                        )
                    else:
                        self.buffer.compute_returns_and_advantages(
                            last_value=0.0,
                            gamma=self.config["gamma"],
                            gae_lambda=self.config["gae_lambda"],
                        )
    
    def train(
        self,
        env_func: Callable,
        num_steps: int = 1000,
        num_envs: int = 1,
        total_timesteps: int = 100000,
        log_interval: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train policy using PPO.
        
        Args:
            env_func: Function that returns an environment
            num_steps: Number of steps to collect per iteration
            num_envs: Number of parallel environments
            total_timesteps: Total number of timesteps to train for
            log_interval: Logging interval in updates
            
        Returns:
            Training metrics
        """
        # Initialize metrics
        metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "approx_kl": [],
            "clip_fraction": [],
            "loss": [],
            "explained_variance": [],
            "fps": [],
        }
        
        # Calculate number of updates
        updates = total_timesteps // num_steps
        
        # Training loop
        for update in range(1, updates + 1):
            start_time = time.time()
            
            # Collect rollouts
            self.collect_rollouts(env_func, num_steps, num_envs)
            
            # Get batch
            batch = self.buffer.get_batch(
                normalize_advantages=True,
            )
            
            # Train on batch
            update_metrics = self.train_on_batch(batch)
            
            # Calculate FPS
            end_time = time.time()
            fps = int(num_steps / (end_time - start_time))
            update_metrics["fps"] = fps
            
            # Update metrics history
            for k, v in update_metrics.items():
                metrics_history[k].append(v)
            
            # Log progress
            if update % log_interval == 0:
                logger.info(
                    f"Update {update}/{updates}, "
                    f"Total steps: {update * num_steps}, "
                    f"FPS: {fps}, "
                    f"Policy loss: {update_metrics['policy_loss']:.4f}, "
                    f"Value loss: {update_metrics['value_loss']:.4f}, "
                    f"Entropy: {update_metrics['entropy']:.4f}, "
                    f"KL: {update_metrics['kl_divergence']:.4f}, "
                    f"Explained var: {update_metrics['explained_variance']:.4f}"
                )
                
                # Early stopping based on KL divergence
                if update_metrics["kl_divergence"] > 2.0 * self.config["kl_target"]:
                    logger.info(f"Early stopping due to high KL divergence")
                    break
        
        return metrics_history


class SalesEnvironment:
    """
    Environment for training language models on sales conversations.
    
    This environment simulates sales interactions with:
    - Realistic customer responses based on emotional state
    - Sales objection patterns with varying difficulty
    - Custom reward functions focused on close rates
    
    This is a stub implementation - the full environment would be implemented
    in a separate module with more sophisticated customer simulation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sales environment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            Initial state
        """
        # Reset conversation state
        self.conversation = []
        self.turn = 0
        self.done = False
        
        # Add initial customer message
        initial_messages = [
            "Hi, I'm interested in learning more about your software development services.",
            "Hello, I've been looking for a team to help with our web application.",
            "I need some information about your software consulting rates.",
            "My company is considering outsourcing our software development.",
            "We're in the market for a new development partner.",
        ]
        initial_message = np.random.choice(initial_messages)
        self.conversation.append({"role": "user", "content": initial_message})
        
        # Create initial state
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            (next_state, reward, done, info)
        """
        # Convert action to response
        response = self._action_to_response(action)
        self.conversation.append({"role": "assistant", "content": response})
        
        # Generate customer response
        if not self.done:
            customer_response = self._generate_customer_response(response)
            self.conversation.append({"role": "user", "content": customer_response})
        
        # Compute reward
        reward = self._compute_reward()
        
        # Increment turn counter
        self.turn += 1
        
        # Check if conversation is done
        if self.turn >= 10 or self._is_conversation_complete():
            self.done = True
        
        # Get next state
        next_state = self._get_state()
        
        # Return step results
        return next_state, reward, self.done, {}
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State representation
        """
        # In a real implementation, this would convert the conversation
        # into a proper state representation (e.g., token IDs)
        # For now, return a dummy state
        state = np.zeros(128, dtype=np.float32)
        # Add turn information
        state[0] = self.turn / 10.0
        # Add some content based on messages
        if len(self.conversation) > 0:
            last_msg = self.conversation[-1]["content"]
            # Simple encoding of the message length
            state[1] = min(len(last_msg) / 100.0, 1.0)
        return state
    
    def _action_to_response(self, action: int) -> str:
        """
        Convert action index to response text.
        
        Args:
            action: Action index
            
        Returns:
            Response text
        """
        # In a real implementation, this would decode from vocabulary or generate
        # structured responses. For now, use template responses.
        responses = [
            "I'd be happy to tell you more about our software development services. We specialize in custom application development with a focus on quality and scalability. What specific needs does your project have?",
            "Our team has extensive experience in web application development. We've worked with companies across various industries to build robust, user-friendly applications. Could you share more about your project goals?",
            "Our rates depend on the specific requirements and scope of your project. We typically work on either a time-and-materials basis or fixed-price contracts. Would you prefer to discuss ballpark figures now or after learning more about our services?",
            "Many companies choose to partner with us for their development needs because it allows them to focus on their core business while we handle the technical implementation. What challenges are you facing with your current development approach?",
            "We've been in business for over 8 years and have successfully delivered more than 100 projects for clients ranging from startups to Fortune 500 companies. What prompted you to look for a new development partner?",
        ]
        
        # Add some basic responses about pricing, timeline, and team
        additional_responses = [
            "Our typical projects range from $50,000 to $500,000 depending on complexity and scope. We can provide a detailed estimate after understanding your requirements better.",
            "We can typically begin work within 2-3 weeks of finalizing the contract. The overall timeline will depend on the project scope, but we pride ourselves on meeting deadlines.",
            "Our team consists of experienced developers, designers, and project managers who have worked together on numerous successful projects. Everyone on our team has at least 3 years of professional experience.",
        ]
        
        all_responses = responses + additional_responses
        index = action % len(all_responses)
        return all_responses[index]
    
    def _generate_customer_response(self, assistant_message: str) -> str:
        """
        Generate customer response based on assistant message.
        
        Args:
            assistant_message: Assistant's message
            
        Returns:
            Customer response
        """
        # Simple keyword-based response generation
        # In a real implementation, this would use a more sophisticated model
        
        if "rates" in assistant_message.lower() or "price" in assistant_message.lower():
            responses = [
                "That's helpful. What would be your estimate for a medium-sized e-commerce site?",
                "That's a bit higher than we budgeted. Is there any flexibility on pricing?",
                "That seems reasonable. Do you offer any payment plans or milestone-based payments?",
            ]
        elif "timeline" in assistant_message.lower() or "schedule" in assistant_message.lower():
            responses = [
                "Good to know. We're hoping to launch in the next 6 months. Is that feasible?",
                "That works for our timeline. How do you handle potential delays?",
                "We need this completed sooner. Is there any way to accelerate the development?",
            ]
        elif "team" in assistant_message.lower() or "experience" in assistant_message.lower():
            responses = [
                "Impressive background. Do you have experience in our industry specifically?",
                "Good to hear. Will we have dedicated developers or are they shared across projects?",
                "That's what we're looking for. How do you handle communication during the project?",
            ]
        elif "project" in assistant_message.lower() or "requirements" in assistant_message.lower():
            responses = [
                "We're building a customer portal that needs to integrate with our existing systems.",
                "We have a legacy application that needs modernizing while maintaining existing functionality.",
                "We're looking for a complete redesign and rebuild of our website with e-commerce capabilities.",
            ]
        else:
            # General responses
            responses = [
                "That's interesting. Can you tell me more about your development process?",
                "I appreciate the information. What makes your company different from other development firms?",
                "I'm considering several options. What would you say is your team's greatest strength?",
                "This all sounds promising. What's the next step if we want to move forward?",
            ]
        
        # Check if this might be the final turn
        if self.turn >= 8 or (self.turn >= 5 and np.random.random() < 0.3):
            closing_responses = [
                "This all sounds good. I think we're ready to move forward. Can you send over a proposal?",
                "I need to discuss this with my team, but I'm impressed with what I've heard. I'll be in touch soon.",
                "I think we've covered everything I needed to know. I'd like to schedule a follow-up call with my technical team.",
                "Thanks for the information. I'll need to compare with other vendors and get back to you.",
            ]
            responses.extend(closing_responses)
        
        return np.random.choice(responses)
    
    def _compute_reward(self) -> float:
        """
        Compute reward for the current state.
        
        Returns:
            Reward value
        """
        # Base reward for continuing the conversation
        reward = 0.1
        
        # If conversation is very short, penalize ending
        if self.done and self.turn < 3:
            reward -= 1.0
        
        # Check for positive customer signals in the last response
        if len(self.conversation) > 0 and self.conversation[-1]["role"] == "user":
            last_msg = self.conversation[-1]["content"].lower()
            
            # Positive signals
            positive_phrases = [
                "sounds good", "ready to move forward", "impressed", 
                "proposal", "next step", "pleased", "works for",
                "agree", "makes sense", "reasonable"
            ]
            
            # Check for positive signals

# Check for positive signals
            for phrase in positive_phrases:
                if phrase in last_msg:
                    reward += 0.5
                    break
            
            # Check for closing/conversion signals
            closing_phrases = [
                "send over a proposal", "move forward", "sign", "contract",
                "next steps", "follow-up call", "schedule a meeting"
            ]
            
            for phrase in closing_phrases:
                if phrase in last_msg:
                    reward += 1.0
                    # Bonus for early conversion
                    reward += max(0, (10 - self.turn) * 0.2)
                    break
            
            # Check for negative signals
            negative_phrases = [
                "too expensive", "not what we're looking for", "doesn't work for us",
                "not interested", "looking elsewhere", "too high", "too long"
            ]
            
            for phrase in negative_phrases:
                if phrase in last_msg:
                    reward -= 0.5
                    break
        
        return reward
    
    def _is_conversation_complete(self) -> bool:
        """
        Check if conversation has reached a natural conclusion.
        
        Returns:
            True if conversation is complete
        """
        # Check if the last message indicates completion
        if len(self.conversation) > 0 and self.conversation[-1]["role"] == "user":
            last_msg = self.conversation[-1]["content"].lower()
            
            # Closing phrases
            closing_phrases = [
                "send over a proposal", "move forward", "ready to move forward",
                "i'll be in touch", "follow-up", "get back to you",
                "need to discuss", "schedule a meeting", "not interested"
            ]
            
            for phrase in closing_phrases:
                if phrase in last_msg:
                    return True
        
        return False


def create_sales_environment():
    """
    Factory function to create a sales environment.
    
    Returns:
        SalesEnvironment instance
    """
    return SalesEnvironment()