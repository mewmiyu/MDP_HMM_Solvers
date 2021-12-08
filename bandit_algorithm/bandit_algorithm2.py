import numpy as np
from scipy.special import logsumexp

"""
Bandit algorithm, when using the inference based approach.
You start with an uniform policy and then update the policy, when you sample an action
and get the corresponding reward, according to the Bayes Rule.
"""


class Bandit2:
    def __init__(self, k=10, alpha=.3, action_values=None):
        self.k = k
        # logarithmic form of the uniform policy prior
        self.log_policy = - np.log(self.k) * np.ones(self.k)
        self.action_values = action_values
        self.alpha = alpha
        self.N = np.zeros(k)

        self.total_reward = 0  # count the total reward
        self.avg_reward = []  # count the average reward for every step

    def play(self, n):
        """
        Chooses one action n times and updates the rewards and action-values depending on the chosen action

        :param n: amount of steps
        """
        for _ in range(n):
            action = self.choose_action()
            self.take_action(action)

    def bandit(self, a):
        """
        Returns the corresponding reward to an action

        :param a: action that should be taken
        :return: the corresponding reward
        """
        # actual reward should be negative, takes the normalized action-value of the state as reward
        return (self.action_values[a] - self.action_values.max()) / np.abs(self.action_values.min())

    def choose_action(self):
        """
        Chooses action according to the policy

        :return: the corresponding action
        """
        # choose a random action based on the probabilities of the actions
        # normalize only, when you choose the action for stability
        return np.random.choice(range(self.k), p=np.exp(self.log_policy - logsumexp(self.log_policy)))

    def take_action(self, a):
        """
        Updates the action-value of a chosen action and the corresponding reward

        :param a: chosen action
        """
        self.N[a] = self.N[a] + 1
        reward = self.bandit(a)  # select an action
        # posterior = likelihood * prior
        # log_posterior = likelihood + log_prior
        self.log_policy[a] = self.alpha * reward + self.log_policy[a]

        self.total_reward += reward  # the total reward for every step
        self.avg_reward.append(self.total_reward / sum(self.N))  # average reward

