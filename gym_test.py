import gym

# Create the CartPole-v1 environment
env = gym.make('CartPole-v1')

# Number of episodes to run
num_episodes = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Render the environment
        env.render()

        # Take a random action
        action = env.action_space.sample()

        # Perform the action and get the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Accumulate the total reward
        total_reward += reward

        # Update the current state
        state = next_state

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Close the environment
env.close()