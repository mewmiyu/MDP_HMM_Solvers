import numpy as np
from typing import List, Tuple

from mushroom_rl.core import Agent


class REPSInt(Agent):
    def __init__(self, mdp_info, policy, approximator, learning_rate,
                 features=None):
        self._alpha = learning_rate

        policy.set_q(approximator)
        self.Q = approximator

        self._add_save_attr(_alpha='mushroom', Q='mushroom')

        super().__init__(mdp_info, policy, features)

    def fit(self, dataset: List[List[np.ndarray]]):
        assert len(dataset) == 1
        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state, absorbing)

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

    def _update(self, state, action, reward, next_state, absorbing):
        pass

    def _post_load(self):
        self.policy.set_q(self.Q)
