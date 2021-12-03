import numpy as np


class Bandit:
    def __init__(self, k=10, epsilon=.3, action_values=None):
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.epsilon = epsilon
        self.actions = range(k)
        self.action_values = action_values

        # not my code
        self.total_reward = 0
        self.avg_reward = []

    def bandit(self, a):
        return np.random.randn() + self.action_values[a]

    def take_action(self, a):
        reward = self.bandit(a)
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

        self.total_reward += reward
        self.avg_reward.append(self.total_reward / sum(self.N))

    def choose_action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q)
        return action

    def play(self, n):
        for _ in range(n):
            action = self.choose_action()
            self.take_action(action)

