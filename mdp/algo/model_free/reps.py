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
        policy.set_q(self.Q)
        self.errors = list()
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
        # Bellman error delta_v^i = r_n + max_a Q(s_n', a) - max_a Q(s_n, a)
        error: np.ndarray = reward + np.nanmax(self.Q[next_state, :]) - np.nanmax(self.Q[state, :])
        # self.Q[state, action] = reward + np.max(self.Q[next_state, :]) - np.max(self.Q[state, :])
        print(self.Q[next_state, :])
        self.errors.append(error)

        if absorbing:
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
            #print(eta_optimal)
            # eta_optimal = self.dual_function(eta_start, self.errors, state, action)
            self.Q[state, action] = np.exp((1 / eta_optimal) * np.nanmax(self.errors)) / np.sum(
                self.Q[state, :] * np.exp((1 / eta_optimal) * np.nanmax(self.errors)))
            print("q")
            print(self.Q[state, action])
