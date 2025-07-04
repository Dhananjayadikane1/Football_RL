import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.policy(state)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.value(state)

# PPO Agent
class PPOGoalkeeperAgent:
    def __init__(self, state_dim, action_dim, clip=0.2, lr=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.clip = clip
        self.gamma = 0.99

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(state_tensor).detach().numpy()[0]
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]

    def store_transition(self, transition):
        self.memory.append(transition)

    def learn(self):
        if len(self.memory) < 10:
            return

        states, actions, rewards, next_states, dones, old_probs = zip(*self.memory)
        self.memory = []

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        # Compute critic values and targets
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = targets - values

        # Detach to avoid second backward pass
        targets = targets.detach()
        advantages = advantages.detach()

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss
        probs = self.actor(states)
        dist_probs = probs.gather(1, actions.view(-1, 1)).squeeze()
        ratios = dist_probs / old_probs

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic loss
        critic_loss = nn.MSELoss()(values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        print(f"[Keeper] Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
