import numpy as np
from mushroom_rl.core import MDPInfo

from mushroom_rl.environments.grid_world import AbstractGridWorld
from mushroom_rl.utils import spaces


class DeepSea(AbstractGridWorld):
    def __init__(self, size, goal, start=(0, 0)):
        observation_space = spaces.Discrete(size * size)
        action_space = spaces.Discrete(4)
        horizon = 100
        gamma = .9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        action_mapping = np.random.randint(0, 2, size * size)
        self.action_mapping = np.stack((action_mapping, 1 - action_mapping), -1)

        super().__init__(mdp_info, width=size, height=size, goal=goal, start=start)
        self.num_states = size * size

    def _step(self, state, action):
        self._grid_step(state, action)

        reward = 0.0

        if np.array_equal(state, self._goal):
            reward += 1 + 0.01 / self._width
            absorbing = True

        else:
            reward -= 0.01 / self._width
            absorbing = False

        return state, reward, absorbing, {}
