from typing import Union, List, Tuple

import numpy as np
from mushroom_rl.core import MDPInfo
from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.policy import TDPolicy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.optimize import minimize


def dual_function(eta_array, *args):
    eta = eta_array.item()
    return eta * 0.001 \
           + eta * np.log(np.mean([np.exp(error / eta) for error in args[0][0]]))


class REPS(TD):

    def __init__(self, mdp_info: MDPInfo, policy: TDPolicy, learning_rate: Parameter):
        self.Q = Table(mdp_info.size)
        self.p = Table(mdp_info.size)
        policy.set_q(self.Q)
        self.errors = list()
        super().__init__(mdp_info, policy, self.p, learning_rate)

    @staticmethod
    def _parse(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
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
        self._update(state, action, reward, next_state, absorbing)

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
                absorbing: bool):
        # Bellman error delta_v^i = r_n + max_a Q(s_n', a) - max_a Q(s_n, a)
        error: np.ndarray = reward + np.max(self.p[next_state, :]) - np.max(self.p[state, :])
        self.Q[state, action] = reward + np.max(self.Q[next_state, :]) - np.max(self.Q[state, :])
        self.errors.append(error)

        if absorbing:
            eta_start = np.ones(1)  # Must be larger than 0
            # eta and v are obtained by minimizing the dual function
            result = minimize(
                fun=dual_function,
                x0=eta_start,  # Initial guess
                args=[self.errors],  # Additional arguments for the function
            )
            eta_optimal = result.x.item()
            #eta_optimal = self.dual_function(eta_start, self.errors, state, action)
            self.p[state, action] = np.exp((1 / eta_optimal) * np.max(self.errors)) / np.sum(
                self.Q[state, :] * np.exp((1 / eta_optimal) * np.max(self.errors)))
