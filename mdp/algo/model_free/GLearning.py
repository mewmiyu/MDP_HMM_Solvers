import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.core import MDPInfo
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class GLearning(TD):
    """
    G-Learning algorithm.
    "Taming the Noise in Reinforcement Learning via Soft Updates".
    Roy Fox, Ari Pakman, Naftali Tishby. 2017.
    """

    def __init__(self, mdp_info: MDPInfo, policy: Policy, learning_rate: Parameter, beta_base: float = 0.1,
                 beta_linear: float = 0.1):
        """
        Constructor.

        Args:
            mdp_info: The information about the MDP
            policy: The policy followed by the agent
            learning_rate: The learning rate (alpha)
            beta_base: The base inverse temperature parameter
            beta_linear: The constant for the linear inverse temperature parameter
        """
        self.G = Table(mdp_info.size)
        self.beta_base = beta_base
        self.beta_linear = beta_linear
        self.n_actions = int(mdp_info.size[1])  # use a uniform prior 1/number_of_actions
        self.counter = 0  # to count the time steps in the update

        super().__init__(mdp_info, policy, self.G, learning_rate)

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
                absorbing: bool):
        """
        Updates the state and action values after interaction with the environment in order to find the optimal
        value function G.

        Args:
            state: The current state
            action: The action taken
            reward: The reward obtained
            next_state: The next state
            absorbing: Whether the next state is absorbing or not
        """
        # counter representing time steps
        self.counter += 1
        # beta = k * t, linear as learning increases
        self.beta = self.beta_base + self.beta_linear * self.counter
        # current value of the state, action pair = G(state, action)
        g_current = self.G[state, action]

        # log(sum_action(prior * e^(- beta * G(next_state, action)))
        g_next = logsumexp(np.log(1 / self.n_actions) - self.beta * self.G[next_state, :]) if not absorbing else 0.

        # update rule for G(state, action)
        # G(state, action) = g_current + alpha * (reward - gamma/beta * g_next - g_current)
        self.G[state, action] = g_current + self._alpha(state, action) * (
                reward - (self.mdp_info.gamma / self.beta) * g_next - g_current)
