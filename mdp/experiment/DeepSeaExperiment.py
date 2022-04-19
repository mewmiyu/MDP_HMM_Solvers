from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Agent
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from mdp.algo.model_free.GLearning import GLearning
from mdp.algo.model_free.MIRL import MIRL
from mdp.algo.model_free.PsiLearning import PsiLearning
from mdp.algo.model_free.env.DeepSea import DeepSea
from mdp.experiment.Experiment import Experiment


class DeepSeaExperiment(Experiment):

    def __init__(self, n_episodes: int, k: int, max_workers: int = None, quiet: bool = True):
        """
        Constructor.

        Constructor.

        Args:
            n_episodes: The number of episodes to move the agent
            k: The number of iterations to perform
            max_workers: The maximum number of workers to use
            quiet: If True, the experiment will not print progress bar
        """
        super().__init__(n_episodes, k, max_workers, quiet)

    def run(self):
        max_steps = 3
        steps = list()

        agents = dict(
            q=QLearning,
            psi=PsiLearning,
            g=GLearning,
            mirl=MIRL
        )

        q = [10, 50, 90]

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
                agent = value(env.info, pi, learning_rate)
                reward_k = self.compute_reward(agent, env)
                # q_p10, q_p50, q_p90
                q_p = np.percentile(reward_k, q)
                reward_list = rewards[key]

                for r_i, q_pi in zip(reward_list, q_p):
                    r_i.append(q_pi)

            sum_reward = 0
            for j in range(size - 2):
                sum_reward -= 1 ** j * (0.01 / size)
            best_reward.append(1 + (0.01 / size) + sum_reward)

        steps = np.array(steps)
        for label, marker, alpha, key in zip(labels, markers, alphas, agents.keys()):
            q_p10, q_p50, q_p90 = rewards[key]
            plt.plot(steps, np.array(q_p50), marker=marker, label=label[0])
            plt.fill_between(steps, q_p10, q_p90, label=label[1], alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')
        plt.xlabel('Size of gridworld')
        plt.ylabel(f'Cumulative average reward after {self.n_episodes} episodes')
        plt.title('Deep Sea Experiment - Parallel')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    experiment = DeepSeaExperiment(100, 2, max_workers=1)
    result, time = Experiment.benchmark(experiment.run)
    print(time)
