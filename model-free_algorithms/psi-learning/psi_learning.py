import numpy as np

from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class PsiLearning(TD):
    """
    Psi-Learning algorithm.
    "Approximate Inference and Stochastic Optimal Control".
    Konrad Rawlik, Marc Toussaint, and Sethu Vijayakumar. 2018.
    """
    def __init__(self, mdp_info, policy, learning_rate):
        """
        Initializes the values.

        :param mdp_info: information about the MDP in the experiment.
        :param policy: policy that we want the agent to learn.
        :param learning_rate: the learning rate alpha.
        """
        self.Psi = Table(mdp_info.size)

        super().__init__(mdp_info, policy, self.Psi, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        """
        Updates the state and action values after interaction with the environment in order to find the optimal
        value function Psi

        :param state: current state
        :param action: current action
        :param reward: reward of the action in the state
        :param next_state: next state after executing the action
        :param absorbing: if it's an absorbing state
        """
        # current value of the state, action pair = Psi(state, action)
        psi_current = self.Psi[state, action]

        # mean_Psi(x) = logsumexp_a(Psi(x, a))
        mean_psi_current = logsumexp(self.Psi[state, :]) if not absorbing else 0.
        mean_psi_next = logsumexp(self.Psi[next_state, :]) if not absorbing else 0.

        # update rule for Psi(state, action)
        # Psi(state, action) = psi_current + alpha * (reward + gamma * mean_psi_next - mean_psi_current)
        self.Psi[state, action] = psi_current + (self.mdp_info.gamma * mean_psi_next - mean_psi_current + reward)
