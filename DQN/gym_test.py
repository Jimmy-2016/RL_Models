import gym

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
