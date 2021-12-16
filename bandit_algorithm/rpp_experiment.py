import numpy as np
import matplotlib.pyplot as plt
from rpp import RelativePayoffProcedure
from bandit_algorithm import Bandit


if __name__ == '__main__':
    k = 5
    avg_reward1 = []
    estimated_q = []
    actual_q = []
    policy = []
    action_values_init = np.array(np.random.uniform(low=-10, high=0, size=(k,)))
    # actual reward should be negative, takes the normalized action-value of the state as reward
    action_values = [(action_values_init[a] - np.max(action_values_init)) / np.abs(np.min(action_values_init))
                     for a in range(len(action_values_init))]
    for alpha in ([0.01, .1, 1, 10]):
        bdt = RelativePayoffProcedure(k, alpha, action_values)
        bdt2 = Bandit(k, alpha, action_values, True)
        bdt.play(1000)
        bdt2.play(1000)
        avg_best_reward = bdt2.best_avg_reward
        avg_reward1 = bdt.avg_reward

        plt.plot(avg_reward1, label=f"alpha='{alpha}'")
        plt.xlabel("Steps")
        plt.ylabel(f"Average Reward")
        plt.title(f"{k}-armed Bandit Testbed")
    plt.plot(avg_best_reward, linestyle='-.', label="best reward")
    plt.legend()

    plt.show()
