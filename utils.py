
import gym
import torch
import torch.nn as nn

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


# Create the CartPole environment
env = gym.make('CartPole-v1')

# Run the simulation for a few episodes
for episode in range(3):  # Run for 3 episodes
    total_reward = 0
    done = False
    state = env.reset()  # Reset the environment to get initial state
    while not done:
        # Render the environment (optional, can be commented out)
        env.render()

        # Take a random action
        action = env.action_space.sample()

        # Perform the action and observe the next state, reward, and whether the episode is done
        next_state, reward, done, info, _ = env.step(action)

        # Accumulate total reward
        total_reward += reward

        # Update the current state
        state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Close the environment
env.close()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)