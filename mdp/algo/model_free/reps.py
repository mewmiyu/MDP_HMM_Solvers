from copy import deepcopy

import numpy as np
from mushroom_rl.core import MDPInfo
from mdp.algo.model_free.reps_int import REPSInt
from mushroom_rl.policy import TDPolicy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.optimize import minimize
from scipy.special import logsumexp

# change names of algorithm => not reps, but Psi-REPS
class REPS(REPSInt):

    def __init__(self, mdp_info: MDPInfo, policy: TDPolicy, learning_rate: Parameter, beta_linear: float = .1 / 10,
                 eps=0.0001):
        self.eps = eps
        self.eta = beta_linear
        self.Q = Table(mdp_info.size)
        self.policy_table_base = Table(mdp_info.size)
        policy.set_q(self.Q)
        self.errors = np.zeros(mdp_info.size)
        self.states = list()
        super().__init__(mdp_info, policy, self.Q, learning_rate)

    @staticmethod
    def dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, errors = args

        return eta * eps + eta * np.log(np.mean(np.exp(errors / eta)))
        # return eta * self.eps + eta * np.log(np.mean([np.exp(error / eta) for error in args[0][0]]))

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, errors = args

        gradient = (eps + np.log(np.mean(np.exp(errors / eta))) - np.mean(np.exp(errors / eta) * errors)) / \
                   (eta * np.mean(np.exp(errors / eta)))
        return np.array([gradient])

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
                absorbing: bool):
        # current value of the state, action pair = Psi(state, action)
        psi_current = self.Q[state, action]

        # mean_Psi(x) = logsumexp_a(Psi(x, a))
        mean_psi_current = logsumexp(self.Q[state, :]) if not absorbing else 0.
        mean_psi_next = logsumexp(self.Q[next_state, :]) if not absorbing else 0.

        # update rule for Psi(state, action)
        # Psi(state, action) = psi_current + alpha * (reward + gamma * mean_psi_next - mean_psi_current)
        self.Q[state, action] = psi_current + (reward
                                               + self.mdp_info.gamma * mean_psi_next
                                               - mean_psi_current)
        # self.Q[state, action] = reward + np.max(self.Q[next_state, :])
        self.states.append(state)
        # error: np.ndarray = reward + np.nanmax(self.Q[next_state, :]) - np.nanmax(self.Q[state, :])
        # self.errors.append(error)

        if absorbing:
            # compute advantage over state action space
            for state in self.states:
                self.errors[state, :] = self.Q[state, :] - np.max(self.Q[state, :])
            policy_table = deepcopy(self.policy_table_base)

            eta_start = np.array(self.eta)  # Must be larger than 0
            # beta is obtained by minimizing the dual function
            result = minimize(
                fun=self.dual_function,
                x0=eta_start,  # Initial guess
                jac=REPS._dual_function_diff,  # gradient function
                bounds=((np.finfo(np.float32).eps, np.inf),),
                args=(self.eps, self.errors),  # Additional arguments for the function
            )
            eta_optimal = result.x.item()
            for state in self.states:
                policy_table[state, :] = np.exp(eta_optimal * self.errors[state, :]) / (np.sum(
                    self.Q[state, :] * np.exp(eta_optimal * self.errors[state, :])) + 0.0000001)
            self.policy.set_q(policy_table)
