import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithm2 import Bandit2
from bandit_algorithm import Bandit

"""
Comparison of the Bandit Algorithm with and without inference.
"""

if __name__ == '__main__':
    k = 5
    iterations = 200
    avg_reward1 = []
    avg_reward2 = []
    actual_q = []
    estimated_q = []
    estimated_q2 = []
    policy = []
    # compute for various epsilon
    for alpha, epsilon in zip([0.3, 0.5, 1], [0.1, 0.2, 0.3]):
        # do 200 iterations of 1000 steps and compute the average reward
        for i in range(iterations):
            action_values = np.linspace(-10, 0, 5)
            bdt = Bandit(k, epsilon, action_values, True)
            bdt2 = Bandit2(k, alpha, action_values)
            bdt.play(1000)
            bdt2.play(1000)
            if i == 0:
                avg_reward1 = bdt.avg_reward
                avg_reward2 = bdt2.avg_reward
                actual_q = action_values
                estimated_q = bdt.Q
                policy = bdt2.policy
            else:
                avg_reward1 = [x + y for x, y in zip(avg_reward1, bdt.avg_reward)]
                avg_reward2 = [a + b for a, b in zip(avg_reward2, bdt2.avg_reward)]
                actual_q = [x + y for x, y in zip(actual_q, action_values)]
                estimated_q = [x + y for x, y in zip(estimated_q, bdt.Q)]
                policy = [x + y for x, y in zip(policy, bdt2.policy)]

        avg_reward1 = [x/iterations for x in avg_reward1]
        avg_reward2 = [c/iterations for c in avg_reward2]
        actual_q = [x / iterations for x in actual_q]
        estimated_q = [x / iterations for x in estimated_q]
        policy = [x / iterations for x in policy]

        estimated_q2 = [np.exp(alpha * actual_q[x]) * policy[x] for x in range(5)]
        estimated_q2 = [np.log(estimated_q2[i] / sum(estimated_q2)) for i in range(len(estimated_q2))]

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action (greedy)"f'{estimated_q}')
        print("Estimated average value-action (inference)"f'{estimated_q2}')

        plt.plot(avg_reward2, linestyle='--', label=f"alpha='{alpha}'")
        plt.plot(avg_reward1, label=f"epsilon='{epsilon}'")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward for 200 Iterations")
        plt.title("5-armed Bandit Testbed Comparison")
        plt.legend()

    plt.show()
