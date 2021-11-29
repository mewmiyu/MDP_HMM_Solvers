from mushroom_rl.policy import Policy
import numpy as np


class VPolicy(Policy):
    def __init__(self):
        """
        Constructor.
        """
        self._approximator = None
        self._predict_params = dict()

        self._add_save_attr(_approximator='mushroom!',
                            _predict_params='pickle')

    def set_v(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.
        """
        self._approximator = approximator

    def get_v(self):
        """
        Returns:
             The approximator used by the policy.
        """
        return self._approximator

    def draw_action(self, state):
        return np.array([np.random.choice(self._approximator.n_actions)])
