import numpy as np

from mushroom_rl.utils.viewer import Viewer

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces


class CliffWalking(Environment):
    def __init__(self, size, start, goal, p):
        observation_space = spaces.Discrete(size * size)
        action_space = spaces.Discrete(4)
        horizon = 1000
        gamma = 0.9
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        # probability array for taking the action that the agent wants to take
        self.p = [p, 1-p]

        assert not np.array_equal(start, goal)

        assert goal[0] < size and goal[1] < size, \
            'Goal position not suitable for the grid world dimension.'

        self._state = None
        self._height = size
        self._width = size
        self._start = start
        self._goal = goal

        # Visualization
        self._viewer = Viewer(self._width, self._height, 500,
                              self._height * 500 // self._width)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.convert_to_int(self._start, self._width)

        self._state = state

        return self._state

    def render(self):
        for row in range(1, self._height):
            for col in range(1, self._width):
                self._viewer.line(np.array([col, 0]),
                                  np.array([col, self._height]))
                self._viewer.line(np.array([0, row]),
                                  np.array([self._width, row]))

        goal_center = np.array([.5 + self._goal[1],
                                self._height - (.5 + self._goal[0])])
        self._viewer.square(goal_center, 0, 1, (0, 255, 0))

        start_center = np.array([.5 + self._start[1],
                                 self._height - (.5 + self._goal[0])])

        self._viewer.square(start_center, 0, 1, (255, 0, 0))

        state_grid = self.convert_to_grid(self._state, self._width)
        state_center = np.array([.5 + state_grid[1],
                                 self._height - (.5 + state_grid[0])])
        self._viewer.circle(state_center, .4, (0, 0, 255))

        self._viewer.display(.1)

    def _grid_step(self, state, action):
        if np.array_equal(action, [0]):
            # up
            if state[0] > 0:
                state[0] -= 1
        elif np.array_equal(action, [1]):
            # down
            if state[0] + 1 < self._height:
                state[0] += 1
        elif np.array_equal(action, [2]):
            # left
            if state[1] > 0:
                state[1] -= 1
        elif np.array_equal(action, [3]):
            # right
            if state[1] + 1 < self._width:
                state[1] += 1

    def step(self, action):
        state = self.convert_to_grid(self._state, self._width)
        if state[0] > 0 and 0 < state[1] < self._width - 1:
            disturbance = np.random.choice([True, False], p=self.p)
            action = [0] if disturbance else action

        new_state, reward, absorbing, info = self._step(state, action)
        self._state = self.convert_to_int(new_state, self._width)

        return self._state, reward, absorbing, info

    def _step(self, state, action):
        self._grid_step(state, action)

        reward = - 0.5 / self._width
        absorbing = False

        if state[0] == 0 and state[1] > 0:
            # first row, but not last column
            if state[1] < self._width - 1:
                reward = -100.0
                # end experiment, if falls down the cliff
                absorbing = True
            else:
                # reached the goal
                reward += 10 + 0.5 / self._width
                absorbing = True
        return state, reward, absorbing, {}

    @staticmethod
    def convert_to_grid(state, width):
        return np.array([state[0] // width, state[0] % width])

    @staticmethod
    def convert_to_int(state, width):
        return np.array([state[0] * width + state[1]])