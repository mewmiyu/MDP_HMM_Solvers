import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class GLearning(TD):
    """
    G-Learning algorithm.
    "Taming the Noise in Reinforcement Learning via Soft Updates".
    Roy Fox, Ari Pakman, Naftali Tishby. 2017.
    """
    def __init__(self, mdp_info, policy, learning_rate, beta=0.1):
        """
        Initializes the values.

        :param mdp_info: information about the MDP in the experiment.
        :param policy: policy that we want the agent to learn.
        :param learning_rate: the learning rate alpha.
        :param beta: the inverse temperature
        """
        self.G = Table(mdp_info.size)
        self.beta = beta
        self.n_actions = int(mdp_info.size[1])  # use a uniform prior 1/number_of_actions
        self.counter = 0  # to count the time steps in the update

        super().__init__(mdp_info, policy, self.G, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        """
        Updates the state and action values after interaction with the environment in order to find the optimal
        value function G

        :param state: current state
        :param action: current action
        :param reward: reward of the action in the state
        :param next_state: next state after executing the action
        :param absorbing: if it's an absorbing state
        """
        # counter representing time steps
        self.counter += 1
        print(self.counter)
        # beta = k * t, linear as learning increases
        #self.beta = self.beta * self.counter
        # current value of the state, action pair = G(state, action)
        g_current = self.G[state, action]

        # log(sum_action(prior * e^(- beta * G(next_state, action)))
        g_next = logsumexp(np.log(1/self.n_actions) - self.beta * self.G[next_state, :]) if not absorbing else 0.

        # update rule for G(state, action)
        # G(state, action) = g_current + alpha * (reward - gamma/beta * g_next - g_current)
        self.G[state, action] = g_current + self._alpha(state, action) * (
            reward - (self.mdp_info.gamma/self.beta) * g_next - g_current)
