import gym
import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a named tuple for storing experience transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN Agent
class DQNAgent:
    def __init__(self, env, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=1e-3):
        self.env = env
        self.memory = deque(maxlen=10000)  # Replay memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return np.argmax(self.model(torch.tensor(state, dtype=torch.float32)).numpy())

    def remember(self, state, action, reward, next_state):
        self.memory.append(Transition(state, action, next_state, reward))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                self.remember(state, action, reward, next_state)
                state = next_state
                self.replay()
            self.target_train()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Initialize the CartPole-v1 environment
env = gym.make('CartPole-v1')

# Initialize the DQN agent
agent = DQNAgent(env)

# Train the agent
agent.train(episodes=100)
