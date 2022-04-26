from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.core import MDPInfo
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.special import logsumexp


class PsiLearning(TD):
    """
    Psi-Learning algorithm.
    "Approximate Inference and Stochastic Optimal Control".
    Konrad Rawlik, Marc Toussaint, and Sethu Vijayakumar. 2018.
    """

    def __init__(self, mdp_info: MDPInfo, policy: Policy, learning_rate: Parameter,
                 beta_linear: float = 0.01):
        """
        Constructor.

        Args:
            mdp_info: The information about the MDP
            policy: The policy followed by the agent
            learning_rate: The learning rate (alpha)
            beta_linear: The constant for the linear inverse temperature parameter
        """
        self.Psi = Table(mdp_info.size)

        self.beta_linear = beta_linear
        self.counter = 0  # to count the time steps in the update

        super().__init__(mdp_info, policy, self.Psi, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
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
        self.beta = self.beta_linear * self.counter

        # current value of the state, action pair = Psi(state, action)
        psi_current = self.Psi[state, action]

        # mean_Psi(x) = logsumexp_a(Psi(x, a))
        mean_psi_current = logsumexp(self.Psi[state, :]) if not absorbing else 0.
        mean_psi_next = logsumexp(self.Psi[next_state, :]) if not absorbing else 0.

        # update rule for Psi(state, action)
        # Psi(state, action) = psi_current + alpha * (reward/beta + gamma * mean_psi_next - mean_psi_current)
        self.Psi[state, action] = psi_current + (reward/self.beta +
                                                 self.mdp_info.gamma * mean_psi_next
                                                 - mean_psi_current)