import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithm2 import Bandit2
from bandit_algorithm import Bandit

"""
Comparison of the Bandit Algorithm with and without inference.
"""

if __name__ == '__main__':
    k = 5
    avg_best_reward = []
    action_values = np.random.uniform(low=-10, high=0, size=(k,))

    for alpha, epsilon in zip([0.1, 1, 10], [0.01, 0.1, 1]):
        bdt = Bandit(k, epsilon, action_values, True)
        bdt2 = Bandit2(k, alpha, action_values)
        bdt.play(10000)
        bdt2.play(10000)
        avg_reward1 = bdt.avg_reward
        avg_reward2 = bdt2.avg_reward
        actual_q = action_values
        estimated_q = bdt.Q
        policy = bdt2.log_policy
        avg_best_reward = bdt.best_avg_reward

        estimated_q2 = [np.exp(alpha * actual_q[x]) * policy[x] for x in range(len(actual_q))]
        estimated_q2 = [np.log(estimated_q2[i] / sum(estimated_q2)) for i in range(len(estimated_q2))]
        estimated_q2 = [estimated_q2[i] for i in range(len(estimated_q2))]

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action (greedy)"f'{estimated_q}')
        print("Estimated average value-action (inference)"f'{estimated_q2}')

        plt.plot(avg_reward2, linestyle='--', label=f"alpha='{alpha}'")
        plt.plot(avg_reward1, label=f"epsilon='{epsilon}'")

        plt.xlabel("Steps")
        plt.ylabel(f"Average Reward")
        plt.title(f"{k}-armed Bandit Testbed Comparison")
    plt.plot(avg_best_reward, linestyle='-.', label="best reward")
    plt.yscale('symlog')
    plt.tight_layout()
    plt.legend()
    plt.show()
