# RainforcementSnakeGame

# Snake Game with Deep Q-Learning

This project implements a Snake game environment using OpenAI's Gym and Pygame, and trains an agent to play the game using Deep Q-Learning with TensorFlow/Keras.

## Project Structure

```
.
├── dqn_agent.py        # DQN Agent implementation
├── snake_env.py        # Snake Game environment using OpenAI Gym and Pygame
├── train.py            # Training script for the DQN agent
├── model_weights/      # Directory to save trained model weights
└── README.md           # This file
```

## Requirements

- Python 3.x
- numpy
- tensorflow
- gym
- pygame
- matplotlib

You can install the required libraries using pip:

```sh
pip install numpy tensorflow gym pygame matplotlib
```

## Files

### `dqn_agent.py`

This file contains the implementation of the Deep Q-Network (DQN) agent. The agent is responsible for interacting with the environment, remembering past experiences, and learning from them.

### `snake_env.py`

This file defines the Snake game environment using OpenAI's Gym and Pygame. It includes the game's logic, state representation, and rendering using Pygame.

### `train.py`

This is the main training script. It initializes the environment and the DQN agent, then runs the training loop. During training, it displays the game in a Pygame window and periodically saves the model weights.

## How to Run

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/snake-dqn.git
   cd snake-dqn
   ```

2. Install the required libraries:

   ```sh
   pip install numpy tensorflow gym pygame matplotlib
   ```

3. Run the training script:

   ```sh
   python train.py
   ```

During training, the game will be displayed in a Pygame window, allowing you to observe how the agent interacts with the environment. The model weights will be saved periodically in the `model_weights` directory.

## Visualizing Training Progress

The training script also logs the episode rewards and plots the training progress using Matplotlib. After training, a graph of the episode rewards and average rewards over the last 100 episodes will be displayed.

## Notes

- The `train.py` script will save the model weights in the `model_weights` directory every 100 episodes and at the end of training.
- You can adjust the training parameters (number of episodes, batch size, etc.) in the `train.py` script as needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
