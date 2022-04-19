from typing import Callable, Tuple, List, Union

import numpy as np
from mushroom_rl.core import Agent, MDPInfo
from mushroom_rl.distributions import Distribution
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import to_parameter, Parameter
from scipy.optimize import minimize


class REPS(Agent):

    def __init__(self, mdp_info: MDPInfo, distribution: Distribution, policy: Policy, epsilon: Union[float, Parameter],
                 features: Union[object] = None):
        """
        Constructor.

        Args:
            mdp_info: The information about the MDP
            distribution: The distribution to use
            policy: The policy followed by the agent (π0(u|x)=
            epsilon: The KL divergence threshold (maximal information loss)
            features: The features to extract from state (φ(x))
        """
        self._distribution = distribution
        self._epsilon = to_parameter(epsilon)
        self._add_save_attr(_distribution='mushroom', _epsilon='mushroom')
        super().__init__(mdp_info, policy, features)

    @staticmethod
    def bellman_error(state_action_visits: np.ndarray, sum_reward: np.ndarray, sum_features: np.ndarray,
                      v) -> np.ndarray:
        """
        Compute the Bellman error.

        Args:
            state_action_visits: The number of visits to each state-action pair
            sum_reward: The sum of the rewards for each state-action pair
            sum_features: The sum of the features for each state-action pair
            v: TODO Parameter description

        Returns:
            The bellman error.
        """
        return (sum_reward + np.transpose(v) * sum_features) / state_action_visits

    @staticmethod
    def dual_function(eta_array: np.ndarray, *args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the dual function of the REPS algorithm. Parameter format is determined by the minimization process.

        Args:
            eta_array: eta as 1-D array with shape (n,)
            args: The additional arguments which must contain epsilon and the deltas (bellman error)

        Returns:
            TODO Return description
        """
        eta = eta_array.item()
        epsilon, deltas = args
        return eta * epsilon + eta * np.log(np.mean(delta / eta for delta in deltas))

    @staticmethod
    def parametric_policy(eta, deltas, pis):
        """
        TODO Description
        """
        # = Parameter vector
        # How to compute theta k+1 ?
        # pi theta is a distribution?  Page 68
        # Maybe use distribution.log_pdf(theta) instead of np.log(pi)
        return np.argmax(np.sum(np.exp(delta / eta) * np.log(pi) for delta, pi in zip(deltas, pis)))

    def fit(self, dataset: List[List[np.ndarray]]):
        ep_state_action_visits = list()
        ep_sum_rewards = list()
        ep_sum_features = list()

        for sample in dataset:
            state, action, reward, next_state, absorbing = self.parse(sample)
            state_action_visits = self._compute(dataset, state, action)
            sum_reward = self._compute(dataset, state, action, lambda x, u, r, x_dash, a: reward)
            sum_features = self._compute(dataset, state, action, lambda x, u, r, x_dash, a: x_dash - x)
            ep_state_action_visits.append(state_action_visits)
            ep_sum_rewards.append(sum_reward)
            ep_sum_features.append(sum_features)

        self._update(ep_state_action_visits, ep_sum_rewards, ep_sum_features)

    def _update(self, ep_state_action_visits: List[np.ndarray], ep_sum_rewards: List[np.ndarray],
                ep_sum_features: List[np.ndarray]):
        """
        TODO Description
        """
        errors = list()
        for (state_action_visits, sum_reward, sum_features) in zip(ep_state_action_visits, ep_sum_rewards,
                                                                   ep_sum_features):
            v = self._distribution.sample()
            error = REPS.bellman_error(state_action_visits, sum_reward, sum_features, v)
            errors.append(error)

        # Page 66 Model-free Policy Search
        eta_start = np.ones(1)  # Must be larger than 0
        # eta and v are obtained by minimizing the dual function
        result = minimize(
            fun=REPS.dual_function,
            x0=eta_start,  # Initial guess
            args=(self._epsilon(), errors),  # Additional arguments for the function
        )
        eta_optimal = result.x.item()

        # TODO Estimating the New Policy.
        # Reference? https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/Peters2010_REPS.pdf

    def _compute(self, dataset: List[List[np.ndarray]], state_i: np.ndarray, action_i: np.ndarray,
                 func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
                 = lambda state, action, reward, next_state, absorbing: np.ones(1)
                 ) -> np.ndarray:
        """
        Computes the value of a dataset using the given function. Only the samples in the dataset with the same state
        and action are considered.

        Args:
            dataset: The dataset to compute the value
            state_i: The state of the sample
            action_i: The action of the sample
            func: The function to use to compute (map) the value (default: Values are mapped to 1)
        """
        values = list()
        for sample in dataset:
            state, action, reward, next_state, absorbing = self.parse(sample)
            if state_i == state and action_i == action:
                values.append(func(state, action, reward, next_state, absorbing))
        return np.sum(values)

    def parse(self, sample: List[np.ndarray]) -> tuple:
        """
        Parse a sample.

        Args:
            sample: The sample to parse.

        Returns:
            The parsed sample as tuple (state, action, reward, next_state, absorbing).
        """
        state = sample[0]  # x
        action = sample[1]  # u
        reward = sample[2]  # r
        next_state = sample[3]  # x': next state or not?
        absorbing = sample[4]

        if self.phi is not None:
            state = self.phi(state)

        return state, action, reward, next_state, absorbing
