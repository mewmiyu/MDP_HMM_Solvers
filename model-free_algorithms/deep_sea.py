import numpy as np
from mushroom_rl.core import MDPInfo

from mushroom_rl.environments.grid_world import AbstractGridWorld
from mushroom_rl.utils import spaces


class DeepSea(AbstractGridWorld):
    def __init__(self, size, start, goal):
        observation_space = spaces.Discrete(size * size)
        action_space = spaces.Discrete(4)
        horizon = 1000
        gamma = 0.9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info, width=size, height=size, goal=goal, start=start)

    def _step(self, state, action):
        self._grid_step(state, action)

        reward = 0.0
        absorbing = False

        if np.array_equal(state, self._goal):
            reward += 1 + 0.01 / self._width
            absorbing = True

        elif action == 3:
            reward -= 0.01 / self._width

        return state, reward, absorbing, {}
