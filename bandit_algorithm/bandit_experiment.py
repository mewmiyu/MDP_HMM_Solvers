import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithm import Bandit


if __name__ == '__main__':
    k = 5
    iterations = 200
    avg_reward1 = []
    estimated_q = []
    actual_q = []
    for epsilon in ([0.1, 0.2, 0.3]):
        for i in range(iterations):
            np.random.seed(1)
            action_values = [np.random.randn() for _ in range(k)]
            bdt = Bandit(k, epsilon, action_values)
            bdt.play(1000)
            if i == 0:
                avg_reward1 = bdt.avg_reward
                estimated_q = bdt.Q
                actual_q = action_values
            else:
                avg_reward1 = [x + y for x, y in zip(avg_reward1, bdt.avg_reward)]
                estimated_q = [x + y for x, y in zip(estimated_q, bdt.Q)]
                actual_q = [x + y for x, y in zip(actual_q, action_values)]
        avg_reward1 = [x/iterations for x in avg_reward1]
        estimated_q = [x/iterations for x in estimated_q]
        actual_q = [x/iterations for x in actual_q]

        plt.plot(avg_reward1, label=f"epsilon='{epsilon}'")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward for 200 Iterations")
        plt.title("5-armed Bandit Testbed")
        plt.legend()

    plt.show()
