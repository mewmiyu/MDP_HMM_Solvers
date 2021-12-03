import numpy as np


class Bandit:
    def __init__(self, k=10, epsilon=.3, action_values=None):
        """
        Initializes the values for the Bandit algorithm

        :param k: amount of actions
        :param epsilon: value for epsilon-greedy selection
        :param action_values: true action values q*(a)
        """
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.epsilon = epsilon
        self.actions = range(k)
        self.action_values = action_values

        self.total_reward = 0  # count the total reward
        self.avg_reward = []  # count the average reward for every step

    def bandit(self, a):
        """
        Returns the corresponding reward to an action

        :param a: action that should be taken
        :return: the corresponding reward
        """
        # actual reward is selected from a distribution with q*(a) as mean
        return np.random.randn() + self.action_values[a]

    def take_action(self, a):
        """
        Updates the action-value of a chosen action and the corresponding reward

        :param a: chosen action
        """
        reward = self.bandit(a)  # select an action
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

        self.total_reward += reward  # the total reward for every step
        self.avg_reward.append(self.total_reward / sum(self.N))  # average reward
        # we want to use the average reward here because it visualizes how the algorithm improves
        # e.g. the average reward will get greater with every step only if the
        # chosen action was better than the one before

    def choose_action(self):
        """
        Chooses action according to the epsilon-greedy policy

        :return: the corresponding action
        """
        if np.random.uniform(0, 1) <= self.epsilon:  # choose a random value with p(epsilon)
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q)
        return action

    def play(self, n):
        """
        Chooses one action n times and updates the rewards and action-values depending on the chosen action

        :param n: amount of steps
        """
        for _ in range(n):
            action = self.choose_action()
            self.take_action(action)
