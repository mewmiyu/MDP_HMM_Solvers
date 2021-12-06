import numpy as np

"""
Bandit algorithm, when using the inference based approach.
You start with an uniform policy and then update the policy, when you sample an action
and get the corresponding reward, according to the Bayes Rule.
"""


class Bandit2:
    def __init__(self, k=10, alpha=.3, action_values=None):
        self.k = k
        self.policy = np.ones(self.k) / self.k
        self.action_values = action_values
        self.alpha = alpha

        self.total_reward = 0  # count the total reward
        self.avg_reward = []  # count the average reward for every step
        self.total_count = 0  # counts the steps

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
        # actual reward is selected from a distribution with q*(a) as mean
        return np.random.randn() + self.action_values[a]  # needs to be negative

    def choose_action(self):
        """
        Chooses action according to the policy

        :return: the corresponding action
        """
        # choose a random action based on the probabilities of the actions
        return np.random.choice(range(self.k), p=self.policy)

    def take_action(self, a):
        """
        Updates the action-value of a chosen action and the corresponding reward

        :param a: chosen action
        """
        self.total_count = self.total_count + 1
        reward = self.bandit(a)  # select an action
        # posterior = likelihood * prior
        self.policy[a] = np.exp(self.alpha * reward) * self.policy[a]
        self.policy = self.policy / np.sum(self.policy)

        self.total_reward += reward  # the total reward for every step
        self.avg_reward.append(self.total_reward / self.total_count)  # average reward
