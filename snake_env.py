from typing import Tuple
from collections import deque
import random
import gymnasium as gym
import numpy as np


class SnakeEnv(gym.Env):
    def __init__(self, grid_size: int = 20):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = grid_size * 100
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.grid_size // 2
        self.snake = deque([(c, c), (c - 1, c), (c - 2, c)])
        self.direction = (1, 0)
        self.food = self._place_food()
        self.score = self.steps = 0
        return self._get_obs(), {}

    def _place_food(self) -> Tuple[int, int]:
        empty = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                 if (x, y) not in self.snake]
        return random.choice(empty) if empty else (0, 0)

    def _get_obs(self) -> np.ndarray:
        head, tail = self.snake[0], self.snake[-1]
        hx, hy, tx, ty = *head, *tail
        fx, fy = self.food
        gs = self.grid_size
        d = self.direction

        def is_blocked(dx, dy):
            nx, ny = hx + dx, hy + dy
            return nx < 0 or nx >= gs or ny < 0 or ny >= gs or (nx, ny) in self.snake

        # Danger features
        danger = [is_blocked(*d), is_blocked(d[1], -d[0]), is_blocked(-d[1], d[0])]

        # Direction
        dir_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        direction = [0] * 4
        direction[dir_map.get(d, 3)] = 1

        # Food direction
        food_dir = [0] * 4
        if fy < hy: food_dir[0] = 1
        elif fy > hy: food_dir[1] = 1
        if fx < hx: food_dir[2] = 1
        elif fx > hx: food_dir[3] = 1

        # Position + length
        max_len = (gs * gs) / 2
        position = [hx / gs, hy / gs, tx / gs, ty / gs, len(self.snake) / max_len]

        return np.array(danger + direction + food_dir + position, dtype=np.float32)

    def step(self, action: int):
        self.steps += 1
        head = self.snake[0]
        old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        if action == 1:
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:
            self.direction = (self.direction[1], -self.direction[0])

        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self._is_collision(new_head):
            return self._get_obs(), -25.0, True, False, {"score": self.score}

        self.snake.appendleft(new_head)
        reward = 0.0

        if new_head == self.food:
            self.score += 1
            reward = 20.0
            self.food = self._place_food()
        else:
            self.snake.pop()
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = -0.1 + (0.3 if new_dist < old_dist else -0.3)

        truncated = self.steps >= self.max_steps
        if truncated:
            reward -= 10.0

        return self._get_obs(), reward, False, truncated, {"score": self.score}

    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or pos in self.snake