import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithm2 import Bandit2
from bandit_algorithm import Bandit


if __name__ == '__main__':
    k = 5
    avg_reward1 = []
    estimated_q = []
    actual_q = []
    policy = []
    action_values = np.random.uniform(low=-10, high=0, size=(k,))
    for alpha in ([0.01, .1, 1, 10]):
        bdt = Bandit2(k, alpha, action_values)
        bdt2 = Bandit(k, alpha, action_values, True)
        bdt.play(1000)
        bdt2.play(1000)
        avg_best_reward = bdt2.best_avg_reward
        avg_reward1 = bdt.avg_reward
        actual_q = action_values
        policy = bdt.log_policy

        estimated_q = [np.exp(alpha * actual_q[x]) * policy[x] for x in range(len(actual_q))]
        estimated_q = [np.log((estimated_q[i]) / sum(estimated_q)) for i in range(len(estimated_q))]
        estimated_q = [estimated_q[i] for i in range(len(estimated_q))]

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action"f'{estimated_q}')

        plt.plot(avg_reward1, label=f"alpha='{alpha}'")
        plt.xlabel("Steps")
        plt.ylabel(f"Average Reward")
        plt.title(f"{k}-armed Bandit Testbed Comparison")
    plt.plot(avg_best_reward, linestyle='-.', label="best reward")
    plt.legend()

    plt.show()
