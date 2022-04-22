from typing import Union, List, Tuple

import numpy as np
from mushroom_rl.core import Agent, MDPInfo
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.optimize import minimize


class REPS(Agent):

    def __init__(self, mdp_info: MDPInfo, policy: Policy, epsilon: Union[float, Parameter]):
        self.Q = Table(mdp_info.size)
        self.epsilon = epsilon
        super().__init__(mdp_info, policy, self.Q)

    @staticmethod
    def _parse(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse a sample from a dataset.

        Args:
            dataset: The dataset containing the sample to parse.

        Returns:
            The parsed sample as tuple (state, action, reward, next_state, absorbing).
        """
        assert len(dataset) == 1
        sample = dataset[0]
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]

        return state, action, reward, next_state, absorbing

    def fit(self, dataset: List[List[np.ndarray]]):
        assert len(dataset) == 1
        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state)

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray):
        # Bellman error delta_v^i = r_n + max_a Q(s_n', a) - max_a Q(s_n, a)
        error: np.ndarray = reward + np.max(self.Q[next_state, :]) - np.max(self.Q[state, :])

        eta_start = np.ones(1)  # Must be larger than 0
        # eta and v are obtained by minimizing the dual function
        result = minimize(
            fun=self.dual_function,
            x0=eta_start,  # Initial guess
            args=[error],  # Additional arguments for the function
        )
        eta_optimal = result.x.item()

    def dual_function(self, eta_array, *args):
        eta = eta_array.item()
        error: np.ndarray = args
        return eta * self.epsilon + eta * np.log(np.exp(error / eta))