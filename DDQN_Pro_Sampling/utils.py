
import gym
import torch
import torch.nn as nn
import numpy as np

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


# # Create the CartPole environment
# env = gym.make('CartPole-v1')
#
# # Run the simulation for a few episodes
# for episode in range(3):  # Run for 3 episodes
#     total_reward = 0
#     done = False
#     state = env.reset()  # Reset the environment to get initial state
#     while not done:
#         # Render the environment (optional, can be commented out)
#         env.render()
#
#         # Take a random action
#         action = env.action_space.sample()
#
#         # Perform the action and observe the next state, reward, and whether the episode is done
#         next_state, reward, done, info, _ = env.step(action)
#
#         # Accumulate total reward
#         total_reward += reward
#
#         # Update the current state
#         state = next_state
#
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")
#
# # Close the environment
# env.close()


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_decay=0.999, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay
        self.epsilon = epsilon
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.memory)

    def add(self, experience, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # Replace the oldest experience if capacity is reached
            self.memory[self._get_index_to_replace()] = experience
        self.priorities[self._get_index_to_replace()] = priority

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.memory)]
        priorities = priorities ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=True)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(self.beta_end, self.beta * self.beta_decay)

        return indices, samples, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + self.epsilon) ** self.alpha

    def _get_index_to_replace(self):
        return len(self.memory) - 1 if len(self.memory) < self.capacity else random.randint(0, self.capacity - 1)





class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 300)
        self.layer2 = nn.Linear(300, 300)
        self.layer3 = nn.Linear(300, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
