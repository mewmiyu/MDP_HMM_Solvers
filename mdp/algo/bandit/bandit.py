import numpy as np


class Bandit:
    """
    Bandit algorithm.
    "Reinforcement Learning"
    Richard S. Sutton et Andrew G. Barto. 2018
    """

    def __init__(self, k: int = 10, epsilon: float = .3, action_values=None,
                 comparison: bool = False):
        """
        Constructor.

        Args:
            k: The amount of actions
            epsilon: The value for epsilon-greedy selection
            action_values: The true action values q*(a)
        """
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.epsilon = epsilon
        self.actions = range(k)
        self.action_values = action_values
        self.comparison = comparison

        self.total_reward = 0  # count the total reward
        self.avg_reward = []  # count the average reward for every step
        self.best_avg_reward = []  # average reward, when taking the best action for comparison

    def bandit(self, a: int):
        """
        Returns the corresponding reward to an action.

        Args:
            a: The action that should be taken

        Returns:
        The corresponding reward to the action.
        """
        if self.comparison:
            # takes the normalized action-value of the state as reward for the comparison
            return self.action_values[a]
        else:
            # actual reward is selected from a distribution with q*(a) as mean
            return np.random.randn() + self.action_values[a]

    def take_action(self, a):
        """
        Updates the action-value of a chosen action and the corresponding reward

        Args:
            a: The action that should be taken
        """
        reward = self.bandit(a)  # select an action
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + (1 / self.N[a]) * (reward - self.Q[a])

        self.total_reward += reward  # the total reward for every step
        self.avg_reward.append(self.total_reward / sum(self.N))  # average reward
        # we want to use the average reward here because it visualizes how the algorithm improves
        # e.g. the average reward will get greater with every step only if the
        # chosen action was better than the one before
        if self.comparison:
            # average reward, when taking the best action for comparison
            best_action = np.argmax(self.action_values)
            self.best_avg_reward.append((self.action_values[best_action] - np.max(self.action_values))
                                        / np.abs(np.min(self.action_values)))

    def choose_action(self):
        """
        Chooses action according to the epsilon-greedy policy.

        Returns:
            The corresponding action
        """
        if np.random.uniform(0, 1) <= self.epsilon:  # choose a random value with p(epsilon)
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q)
        return action

    def play(self, n: int):
        """
        Chooses one action n times and updates the rewards and action-values depending on the chosen action.

        Args:
            n: The amount of steps
        """
        for _ in range(n):
            action = self.choose_action()
            self.take_action(action)
