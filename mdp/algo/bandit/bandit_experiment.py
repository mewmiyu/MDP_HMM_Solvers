import matplotlib.pyplot as plt
import numpy as np

from Bandit import Bandit

if __name__ == '__main__':
    k = 5
    avg_reward1 = []
    estimated_q = []
    actual_q = []
    action_values = np.random.uniform(low=-10, high=0, size=(k,))
    for epsilon in ([0.01, .1, 1, 10]):
        bdt = Bandit(k, epsilon, action_values)
        bdt.play(1000)
        avg_best_reward = bdt.best_avg_reward
        avg_reward1 = bdt.avg_reward
        estimated_q = bdt.Q
        actual_q = action_values

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action"f'{estimated_q}')

        plt.plot(avg_reward1, label=f"epsilon='{epsilon}'")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.title("5-armed Bandit Testbed")
    plt.plot(avg_best_reward, linestyle='-.', label="best reward")
    plt.legend()

    plt.show()
