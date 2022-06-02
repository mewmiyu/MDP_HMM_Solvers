from copy import deepcopy

import numpy as np
from mushroom_rl.core import MDPInfo
from mdp.algo.model_free.reps_int import REPSInt
from mushroom_rl.policy import TDPolicy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.table import Table
from scipy.optimize import minimize


class REPS(REPSInt):

    def __init__(self, mdp_info: MDPInfo, policy: TDPolicy, learning_rate: Parameter, eps=0.7):
        self.eps = eps
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

        max_error = np.nanmax(errors)
        r = errors - max_error
        sum1 = np.mean(np.exp(r / eta))
        return eta * eps + eta * np.log(sum1) + max_error
        # return eta * self.eps + eta * np.log(np.mean([np.exp(error / eta) for error in args[0][0]]))

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, errors = args

        max_error = np.nanmax(errors)
        r = errors - max_error
        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)
        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)
        return np.array([gradient])

    def _update(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
                absorbing: bool):
        # Use SARSA for updating the q-table
        q_current = self.Q[state, action]

        self.next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

        self.Q[state, action] = q_current + self._alpha(state, action) * (
                reward + self.mdp_info.gamma * q_next - q_current)

        self.states.append(state)
        #error: np.ndarray = reward + np.nanmax(self.Q[next_state, :]) - np.nanmax(self.Q[state, :])
        #self.errors.append(error)

        if absorbing:
            # compute advantage over state action space
            for state in self.states:
                self.errors[state, :] = self.Q[state, :] - np.max(self.Q[state, :])
            policy_table = deepcopy(self.policy_table_base)

            eta_start = np.ones(1)  # Must be larger than 0
            # eta and v are obtained by minimizing the dual function
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
                    np.exp(eta_optimal * self.errors[state, :])) + 0.0000001)
            self.policy.set_q(policy_table)
