import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithm2 import Bandit2


if __name__ == '__main__':
    k = 5
    iterations = 200
    avg_reward1 = []
    estimated_q = []
    actual_q = []
    policy = []
    # compute for various epsilon
    for alpha in ([0.0, 0.1, 1.0]):
        # do 200 iterations of 1000 steps and compute the average reward
        bdt = None
        for i in range(iterations):
            np.random.seed(1)
            action_values = [np.random.randn() - 2 for _ in range(k)]
            bdt = Bandit2(k, alpha, action_values)
            bdt.play(1000)
            if i == 0:
                avg_reward1 = bdt.avg_reward
                actual_q = action_values
                policy = bdt.policy
            else:
                avg_reward1 = [x + y for x, y in zip(avg_reward1, bdt.avg_reward)]
                actual_q = [x + y for x, y in zip(actual_q, action_values)]
                policy = [x + y for x, y in zip(policy, bdt.policy)]
        avg_reward1 = [x/iterations for x in avg_reward1]
        actual_q = [x / iterations for x in actual_q]
        policy = [x / iterations for x in policy]

        estimated_q = [np.exp(alpha * actual_q[x]) * policy[x] for x in range(len(actual_q))]
        estimated_q = [np.log(estimated_q[i] / sum(estimated_q)) for i in range(len(estimated_q))]

        print("Actual average value-action"f'{actual_q}')
        print("Estimated average value-action"f'{estimated_q}')

        plt.plot(avg_reward1, label=f"alpha='{alpha}'")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward for 200 Iterations")
        plt.title("5-armed Bandit Testbed")
        plt.legend()

    plt.show()