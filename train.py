import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import os

# Create a directory to save model weights
if not os.path.exists('model_weights'):
    os.makedirs('model_weights')

# Initialize environment and agent
env = SnakeEnv()
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)

# Training parameters
episodes = 1000
batch_size = 32
update_target_interval = 10  # Update target model every 10 episodes

# Lists to store episode rewards and losses for visualization
episode_rewards = []
average_rewards = []

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, *state_shape])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    episode_rewards.append(total_reward)
    average_rewards.append(np.mean(episode_rewards[-100:]))  # Average reward of the last 100 episodes

    # Update target model periodically
    if e % update_target_interval == 0:
        agent.update_target_model()
        print(f"Target model updated at episode {e}")

    # Save the model periodically
    if e % 100 == 0:
        agent.save(f"model_weights/dqn_model_episode_{e}.weights.h5")

# Save final model
agent.save("model_weights/dqn_model_final.weights.h5")

# Plot rewards
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(average_rewards, label='Average Reward (last 100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title('Training Progress')
plt.show()
