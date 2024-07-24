import gym
import numpy as np
import random
from gym import spaces

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.grid_size = 20
        self.grid_width = 20
        self.grid_height = 20

        # Action space: 0 - left, 1 - right, 2 - up, 3 - down
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_width, self.grid_height, 3), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.snake = [[5, 5]]
        self.food = [random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)]
        self.score = 0
        self.done = False
        self.direction = random.choice([0, 1, 2, 3])
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.grid_width, self.grid_height, 3), dtype=np.uint8)
        for segment in self.snake:
            obs[segment[0], segment[1]] = [0, 255, 0]
        obs[self.food[0], self.food[1]] = [255, 0, 0]
        return obs

    def step(self, action):
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3

        head = self.snake[0][:]
        if self.direction == 0:
            head[1] -= 1
        elif self.direction == 1:
            head[1] += 1
        elif self.direction == 2:
            head[0] -= 1
        elif self.direction == 3:
            head[0] += 1

        if head[0] < 0 or head[0] >= self.grid_width or head[1] < 0 or head[1] >= self.grid_height or head in self.snake:
            self.done = True
            reward = -10
        else:
            self.snake.insert(0, head)
            if head == self.food:
                self.food = [random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)]
                reward = 10
                self.score += 1
            else:
                self.snake.pop()
                reward = 1

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
