import numpy as np
import matplotlib.pyplot as plt
from rpp import RelativePayoffProcedure
from bandit_algorithm import Bandit

"""
Comparison of the Bandit Algorithm with and without inference.
"""

if __name__ == '__main__':
    k = 5
    iterations = 20
    avg_reward1 = []
    avg_reward2 = []
    actual_q = []
    estimated_q = []
    policy = []
    avg_best_reward = []
    action_values_init = np.random.uniform(low=-10, high=0, size=(k,))
    # actual reward should be negative, takes the normalized action-value of the state as reward
    action_values = [(action_values_init[a] - np.max(action_values_init)) / np.abs(np.min(action_values_init))
                     for a in range(len(action_values_init))]
    for alpha, epsilon in zip([0.1, 1, 10], [0.01, 0.1, 1]):
        for i in range(iterations):
            bdt = Bandit(k, epsilon, action_values, True)
            bdt2 = RelativePayoffProcedure(k, alpha, action_values)
            bdt.play(5000)
            bdt2.play(5000)
            if i == 0:
                avg_reward1 = bdt.avg_reward
                avg_reward2 = bdt2.avg_reward
                actual_q = action_values
                estimated_q = bdt.Q
                policy = bdt2.log_policy
                avg_best_reward = bdt.best_avg_reward
            else:
                avg_reward1 = [x + y for x, y in zip(avg_reward1, bdt.avg_reward)]
                avg_reward2 = [x + y for x, y in zip(avg_reward2, bdt2.avg_reward)]
                actual_q = [x + y for x, y in zip(actual_q, action_values)]
                estimated_q = [x + y for x, y in zip(estimated_q, bdt.Q)]
                policy = [x + y for x, y in zip(policy, bdt2.log_policy)]
                avg_best_reward = [x + y for x, y in zip(avg_best_reward, bdt.best_avg_reward)]
        avg_reward1 = [x/iterations for x in avg_reward1]
        avg_reward2 = [x/iterations for x in avg_reward2]
        actual_q = [x/iterations for x in actual_q]
        estimated_q = [x/iterations for x in estimated_q]
        policy = [x/iterations for x in policy]
        avg_best_reward = [x/iterations for x in avg_best_reward]

        estimated_q2 = [(np.exp(alpha * actual_q[x] * policy[x])) / sum(actual_q) for x in range(len(actual_q))]
        #estimated_q2 = [np.log(estimated_q2[i]) for i in range(len(estimated_q2))]

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action (greedy)"f'{estimated_q}')
        print("Estimated average value-action (inference)"f'{estimated_q2}')

        plt.plot(avg_reward2, linestyle='--', label=f"alpha='{alpha}'")
        plt.plot(avg_reward1, label=f"epsilon='{epsilon}'")

        plt.xlabel("Steps")
        plt.ylabel(f"Average Symlog Reward for {iterations} iterations")
        plt.title(f"{k}-armed Bandit Testbed Comparison")
    plt.plot(avg_best_reward, linestyle='-.', label="best reward")
    plt.yscale('symlog', linthresh=0.01)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()
