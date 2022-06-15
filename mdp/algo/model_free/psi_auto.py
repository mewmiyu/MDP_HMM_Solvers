import numpy as np
from mushroom_rl.core import MDPInfo
from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.policy import TDPolicy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.optimize import minimize
from scipy.special import logsumexp


class PsiAuto(TD):
    """
            Automatically beta-tuning for the Psi-algorithm.

            Based on:
            "Approximate Inference and Stochastic Optimal Control".
            Konrad Rawlik, Marc Toussaint, and Sethu Vijayakumar. 2018.

            And the dual function minimization idea is from:
            "Relative Entropy Policy Search"
            Jan Peters, Katharina Muelling, Yasemin Altun. 2012
    """

    def __init__(self, mdp_info: MDPInfo, policy: TDPolicy, learning_rate: Parameter, beta_linear: float = 0.01,
                 eps=1):
        """
        Constructor.

        Args:
            mdp_info: The information about the MDP
            policy: The policy followed by the agent
            learning_rate: The learning rate (alpha)
            beta_linear: The constant for the linear inverse temperature parameter
            eps: the maximum admissible value for the Kullback-Leibler divergence between
                 the new inverse temperature and the previous one at each update step.
        """
        self.eps = eps
        self.beta = beta_linear
        self.Q = Table(mdp_info.size)
        policy.set_q(self.Q)
        self.errors = np.zeros(mdp_info.size)
        self.states = list()
        self.betas = list()
        super().__init__(mdp_info, policy, self.Q, learning_rate)

    @staticmethod
    def dual_function(beta_array, *args):
        """
        Computes the dual function

        Args:
        beta_array: value for the inverse temperature beta
        args: additional arguments for the dual function
        """
        beta = (1 / beta_array.item())
        eps, errors = args

        return beta * eps + beta * np.log(np.mean(np.exp(errors / beta)))

    @staticmethod
    def _dual_function_diff(beta_array, *args):
        """
            Computes the dual function derivative

            Args:
            beta_array: value for the inverse temperature beta
            args: additional arguments for the dual function
        """
        beta = (1 / beta_array.item())
        eps, errors = args

        gradient = (eps + np.log(np.mean(np.exp(errors / beta))) - np.mean(np.exp(errors / beta) * errors)) / \
                   (beta * np.mean(np.exp(errors / beta)))
        return np.array([gradient])

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
        # current value of the state, action pair = Psi(state, action)
        psi_current = self.Q[state, action]

        # mean_Psi(x) = logsumexp_a(Psi(x, a))
        mean_psi_current = logsumexp(self.Q[state, :]) if not absorbing else 0.
        mean_psi_next = logsumexp(self.Q[next_state, :]) if not absorbing else 0.

        # update rule for Psi(state, action)
        # Psi(state, action) = psi_current + alpha * (reward/beta + gamma * mean_psi_next - mean_psi_current)
        self.Q[state, action] = psi_current + (reward / self.beta +
                                               self.mdp_info.gamma * mean_psi_next
                                               - mean_psi_current)
        # save the states for iterating through them at the end of the episode
        self.states.append(state)

        if absorbing:  # last state
            # compute advantage over state action space
            for state in self.states:
                self.errors[state, :] = self.Q[state, :] - np.max(self.Q[state, :])

            eta_start = np.array(self.beta)  # Start with an initial guess for the inverse temperature
            # beta is obtained by minimizing the dual function
            result = minimize(
                fun=self.dual_function,
                x0=eta_start,  # Initial guess
                jac=self._dual_function_diff,  # gradient function
                bounds=((np.finfo(np.float32).eps, np.inf),),
                args=(self.eps, self.errors),  # Additional arguments for the function
            )
            self.beta = result.x.item()
            self.betas.append(self.beta)
            for state in self.states:
                # current value of the state, action pair = Psi(state, action)
                psi_current = self.Q[state, action]

                # mean_Psi(x) = logsumexp_a(Psi(x, a))
                mean_psi_current = logsumexp(self.Q[state, :]) if not absorbing else 0.
                mean_psi_next = logsumexp(self.Q[next_state, :]) if not absorbing else 0.

                # update rule for Psi(state, action)
                # Psi(state, action) = psi_current + alpha * (reward/beta + gamma * mean_psi_next - mean_psi_current)
                self.Q[state, action] = psi_current + (reward / self.beta +
                                                       self.mdp_info.gamma * mean_psi_next
                                                       - mean_psi_current)
