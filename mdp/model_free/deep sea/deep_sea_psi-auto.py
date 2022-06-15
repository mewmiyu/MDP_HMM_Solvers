from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.algorithms.value.td.q_learning import QLearning
from mdp.algo.model_free.psi_learning import PsiLearning
from mushroom_rl.core import Core, Agent, Environment
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from mdp.algo.model_free.env.deep_sea import DeepSea
from mdp.algo.model_free.psi_auto import PsiAuto
from mdp.experiment.model_free import Experiment


def experiment_deepsea(agent: Agent, env: Environment, n_episodes: int, k: int) -> List[np.ndarray]:
    reward_k = list()
    for seed in range(k):
        # Set the seed
        np.random.seed(seed)

        # Reinforcement learning experiment
        core = Core(agent, env)
        # Train
        core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False, quiet=True)
        # Evaluate results for n_episodes
        dataset_q = core.evaluate(n_episodes=1, render=False, quiet=True)
        # Compute the average objective value
        r = np.mean(compute_J(dataset_q, 1))
        reward_k.append(r)
    return reward_k


def run():
    max_steps = 7
    steps = list()

    k = 25
    n_episodes = 100

    agents = dict(
        psia=PsiAuto,
        psi=PsiLearning,
        q=QLearning
    )

    q = [25, 50, 75]

    betas = list()

    labels: map[List[str]] = map(lambda l: [f'{l}_median', f'{l}_10:90'], agents.keys())
    markers = ['o', '^', '>', '<']
    alphas = [.3, .25, .2, .15]

    rewards: Dict[str, List[List[np.ndarray]]] = dict()
    for key in agents.keys():
        l_q = list()
        for _ in q:
            l_q.append(list())
        rewards[key] = l_q

    best_reward = list()

    for exponent in range(1, max_steps + 1):
        size = np.power(2, exponent)
        steps.append(size)
        print('Step: {}, size: {}'.format(exponent, size))

        # Create the grid environment
        env = DeepSea(size, start=(0, 0), goal=(size - 1, size - 1))

        # Use an epsilon-greedy policy
        epsilon = .1
        pi = EpsGreedy(epsilon=epsilon)

        learning_rate = Parameter(.1 / 10)

        for key, value in agents.items():
            agent = value(env.info, pi, learning_rate=learning_rate)
            reward_k = experiment_deepsea(agent, env, n_episodes, k)
            # q_p10, q_p50, q_p90
            q_p = np.percentile(reward_k, q)

            reward_list = rewards[key]
            for r_i, q_pi in zip(reward_list, q_p):
                r_i.append(q_pi)

            if key == 'psia':
                betas.append(np.mean(agent.betas))

        sum_reward = 0
        for j in range(size - 2):
            sum_reward -= 1 ** j * (0.01 / size)
        best_reward.append(1 + (0.01 / size) + sum_reward)

    steps = np.array(steps)
    for label, marker, alpha, key in zip(labels, markers, alphas, agents.keys()):
        q_p10, q_p50, q_p90 = rewards[key]
        plt.plot(steps, np.array(q_p50), marker=marker, label=label[0])
        plt.fill_between(steps, q_p10, q_p90, alpha=alpha)

    plt.plot(steps, best_reward, label='Best reward')
    plt.xlabel('Size of gridworld')
    plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
    plt.title('Deep Sea Experiment')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    plt.plot(steps, betas, label='Inverse Temperature', marker='o')
    plt.xlabel('Size of gridworld')
    plt.ylabel(f'Value of the inverse temperature after {n_episodes} episodes')
    plt.title('Deep Sea Experiment')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    result, time = Experiment.benchmark(run)
    print(time)
