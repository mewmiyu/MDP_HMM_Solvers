import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class MIRL(TD):
    """
        MIRL algorithm.
        "Soft Q-Learning With Mutual-Information Regularization".
        Jordi Grau-Moya, Felix Leibfried and Peter Vrancx. 2019.
    """

    def __init__(self, mdp_info, policy, learning_rate, beta_base=0.1, beta_linear=0.1):
        """
        Initializes the values.

        :param mdp_info: information about the MDP in the experiment.
        :param policy: policy that we want the agent to learn.
        :param learning_rate: the learning rate alpha.
        """
        self.Q = Table(mdp_info.size)
        self.policy = policy
        self.prior = np.ones(int(mdp_info.size[1])) * (1 / int(mdp_info.size[1]))  # use a uniform prior
        # uses the same equation for beta as g-learning
        self.beta_base = beta_base
        self.beta_linear = beta_linear
        self.counter = 0  # to count the time steps in the update

        super().__init__(mdp_info, policy, self.Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        """
        Updates the state and action values after interaction with the environment in order to find the optimal
        value function Q

        :param state: current state
        :param action: current action
        :param reward: reward of the action in the state
        :param next_state: next state after executing the action
        :param absorbing: if it's an absorbing state
        """
        # counter representing time steps
        self.counter += 1
        # beta = k * t, linear as learning increases
        self.beta = self.beta_base + self.beta_linear * self.counter

        # current value of the state, action pair = Q(state, action)
        q_current = self.Q[state, action]

        # update the prior
        # p_i+1(a) = (1-alpha)p_i(a) + alpha * policy_i(a|s_i)
        self.prior[action] = (1 - self._alpha(state, action)) * self.prior[action] \
                             + self._alpha(state, action) * self.policy(state, action)

        # compute the empirical soft-operator
        # r(s, a) + gamma/ beta logsumexp(log(prior(a') + exp(beta Q(s,a'))))
        t_emp = reward + (self.mdp_info.gamma / self.beta) * \
                logsumexp(np.log(self.prior) + self.beta * self.Q[next_state, :]) if not absorbing else 0.

        # update rule for G(state, action)
        # Q(state, action) = q_current + alpha * (t_emp - q_current)
        self.Q[state, action] = q_current + self._alpha(state, action) * (
                t_emp - q_current)
