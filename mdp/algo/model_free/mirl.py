import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.core import MDPInfo
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class MIRL(TD):
    """
        MIRL algorithm.
        "Soft Q-Learning With Mutual-Information Regularization".
        Jordi Grau-Moya, Felix Leibfried and Peter Vrancx. 2019.
    """

    def __init__(self, mdp_info: MDPInfo, policy: Policy, learning_rate: Parameter, alpha_prior: float = 0.00000001,
                 beta_linear: float = 0.00000001):
        """
        Constructor.

        Args:
            mdp_info: The information about the MDP
            policy: The policy followed by the agent
            learning_rate: The learning rate (alpha)
            beta_base: The base inverse temperature parameter
            beta_linear: The constant for the linear inverse temperature parameter
        """
        self.Q = Table(mdp_info.size)
        self.prior = np.ones(int(mdp_info.size[1])) * (1 / int(mdp_info.size[1]))  # use a uniform prior

        self.n_actions = int(mdp_info.size[1])  # use a uniform prior 1/number_of_actions
        # uses the same equation for beta as g-learning
        self.alpha_prior = alpha_prior
        self.beta_linear = beta_linear
        self.counter = 0  # to count the time steps in the update

        super().__init__(mdp_info, policy, self.Q, learning_rate)

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
                absorbing: bool):
        """
        Updates the state and action values after interaction with the environment in order to find the optimal
        value function Q.

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
        self.beta = self.beta_linear * self.counter

        # current value of the state, action pair = Q(state, action)
        q_current = self.Q[state, action]

        # update the prior
        # p_i+1(.) = (1-alpha)p_i(.) + alpha * policy_i(.|s_i)
        self.prior = (1 - self.alpha_prior) * self.prior[:] \
                             + self.alpha_prior * np.array([self.policy(state, a) for a in range(4)])

        # compute the empirical soft-operator
        # r(s, a) + gamma/ beta logsumexp(log(prior(a') + exp(beta Q(s,a'))))
        q_next = logsumexp(np.log(self.prior) - self.beta * self.Q[next_state, :]) if not absorbing else 0.

        t_emp = reward - (self.mdp_info.gamma / self.beta) * q_next

        # update rule for G(state, action)
        # Q(state, action) = q_current + alpha * (t_emp - q_current)
        self.Q[state, action] = q_current + self._alpha(state, action) * (
                t_emp - q_current)
