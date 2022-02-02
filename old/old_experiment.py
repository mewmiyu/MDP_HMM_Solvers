import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from mushroom_rl.environments import *
from mushroom_rl.utils.callbacks import CollectQ
from mushroom_rl.utils.parameters import ExponentialParameter

"""
Source for chain_structure: 
https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/double_chain_q_learning/double_chain.py
"""


def experiment(agent_name, agent_q):
    # Reinforcement learning experiment
    collect_Q = CollectQ(agent_q)
    callbacks = [collect_Q]

    core_value = Core(agent_name, env, callbacks)
    core_test = Core(agent_name, env, callbacks)

    j_q = list()

    for i in range(51):
        # Evaluate results for n_steps
        dataset_q = core_value.evaluate(n_steps=10, render=False)
        # Compute the average objective value
        j_q.append(np.mean(compute_J(dataset_q, env.info.gamma)))
        # Train
        core_test.learn(n_steps=300, n_steps_per_fit=1, render=False)
    return j_q


if __name__ == '__main__':
    from mushroom_rl.core import Core
    from mushroom_rl.algorithms.value.td.q_learning import QLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.utils.dataset import compute_J

    all_j_q = list()
    all_j_g = list()

    # MDP
    path = Path(__file__).resolve().parent / 'chain_structure'
    # Source: https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/double_chain_q_learning/double_chain.py
    p = np.load(path / 'p.npy')
    for i in range(p.shape[0]):
        for k in range(p.shape[1]):
            for m in range(p.shape[2]):
                if p[i, k, m] == 0.8:
                    p[i, k, m] = 1.0
                if p[i, k, m] == 0.2:
                    p[i, k, m] = 0.0
    #print(p)
    #p = np.load(path / 'p.npy')
    rew = np.load(path / 'rew.npy')
    mu = np.zeros(p.shape[0])
    mu[0] = 1
    env = FiniteMDP(p, rew, gamma=0.9, mu=mu)

    # Policy
    epsilon = Parameter(value=0.0)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialParameter(value=1., exp=.71, size=env.info.size)
    algorithm_params = dict(learning_rate=learning_rate)

    for k in range(20):
        np.random.seed(k)

        agent_Q = QLearning(env.info, pi, learning_rate=learning_rate)
        agent_v = agent_Q.Q
        j_q = experiment(agent_Q, agent_v)

        env.reset()

        # agent_G = GLearning(env.info, pi, learning_rate=learning_rate,
        #                    beta_linear=100, beta_base=1)
        # agent_v = agent_G.G
        # j_g = experiment(agent_G, agent_v)

        all_j_q.append(j_q)
        # all_j_g.append(j_g)

    all_j_q = np.array(all_j_q)
    # all_j_g = np.array(all_j_g)
    steps = np.arange(0, 510, 10)
    # Compute the 10, 50, 90-th percentiles and plot them
    q_p10, q_p50, q_p90 = np.percentile(all_j_q, [10, 50, 90], 0)
    # g_p10, g_p50, g_p90 = np.percentile(all_j_g, [10, 50, 90], 0)
    plt.fill_between(steps, q_p10, q_p90, where=q_p90 >= q_p10, label='q: [10-90] percentiles', alpha=0.2)
    plt.plot(steps, q_p50, label='q: median')
    # plt.fill_between(steps, g_p10, g_p90, where=g_p90 >= g_p10, label='g: [10-90] percentiles',
    #                 alpha=0.2)
    # plt.plot(steps, g_p50, label='g: median')
    plt.xlabel('steps')
    plt.ylabel('cumulative discounted reward')
    plt.title('Double Chain Experiment: Q-Learning vs G-Learning')
    plt.legend()
    plt.show()
