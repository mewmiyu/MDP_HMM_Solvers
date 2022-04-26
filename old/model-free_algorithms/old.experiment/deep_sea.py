from typing import List, Callable, Union, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Agent
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from mdp.algo.model_free.env.deep_sea import DeepSea
from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.reps import REPS
from mdp.experiment.model_free import ExperimentParser, Experiment


class DeepSeaParser(ExperimentParser):
    """
    Parses the command line arguments for a deep sea old.experiment.
    """

    def __init__(self):
        super().__init__('Deep Sea')
        self._parser.add_argument('-m', '--max_steps', metavar='int', type=int, help='Maximum number of steps')


class DeepSeaExperiment(Experiment):
    """
    The Deep Sea old.experiment.

    Parameters:
        DeepSeaExperiment.AGENTS: A list of available agents
        DeepSeaExperiment.Q: A list of q values for the percentile. The values must be between 0 and 100 inclusive

    """
    AGENTS: Dict[str, Agent] = dict(
        q=QLearning,
        psi=PsiLearning,
        g=GLearning,
        mirl=MIRL,
        reps=REPS
    )

    Q: List[int] = [10, 50, 90]

    def __init__(self, agent: List[Union[str, Callable[..., Agent]]], n_episodes: int, k: int, max_steps: int):
        """
        Constructor.

        Args:
            agent: The agent to use
            n_episodes: The number of episodes to move the agent
            k: The number of iterations to perform
            max_steps: The maximum number of steps to perform
        """
        super().__init__(agent, n_episodes, k)
        self.max_steps = max_steps

    @staticmethod
    def plot(filenames: List[str], alphas: List[float], markers: Tuple[str], *args):
        max_steps, n_episodes = args
        experiment_name = DeepSeaExperiment.__name__

        steps = list()
        best_reward = list()
        for exponent in range(1, max_steps + 1):
            size = np.power(2, exponent)
            steps.append(size)

            sum_reward = 0
            for j in range(size - 2):
                sum_reward -= 1 ** j * (0.01 / size)
            best_reward.append(1 + (0.01 / size) + sum_reward)

        steps = np.array(steps)
        for filename, alpha, marker in zip(filenames, alphas, markers):
            agent_name = filename.replace(f'{experiment_name}_', '').replace('.npy', '')
            q_p10, q_p50, q_p90 = np.load(filename)

            labels = Experiment.create_labels(f'{agent_name}_median', f'{agent_name}_10:90')
            plt.plot(steps, q_p50, marker=marker, label=labels[0])
            plt.fill_between(steps, q_p10, q_p90, label=labels[1], alpha=0.25)

        plt.plot(steps, best_reward, label='Best reward')
        plt.xlabel('Size of gridworld')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        plt.title('Deep Sea Experiment')
        plt.legend()
        plt.show()

    def run(self, write: bool = False) -> List[List[np.ndarray]]:
        q_ps: List[List[np.ndarray]] = list()
        for _ in self.Q:
            q_ps.append(list())

        for exponent in range(1, self.max_steps + 1):
            size = np.power(2, exponent)

            # Use an epsilon-greedy policy
            epsilon = .1
            pi = EpsGreedy(epsilon=epsilon)

            env = DeepSea(size, start=(0, 0), goal=(size - 1, size - 1))
            learning_rate = Parameter(.1 / 10)
            agent = self.agent[1](env.info, pi, learning_rate)

            # q_p10, q_p50, q_p90
            reward_k = self.compute_reward(agent, env, self.n_episodes, self.k)

            for q_pi, q_p in zip(q_ps, np.percentile(reward_k, self.Q)):
                q_pi.append(q_p)

            if write:
                self.save(f'{DeepSeaExperiment.__name__}_{self.agent[0]}', q_ps)
        return q_ps


if __name__ == '__main__':
    parser = DeepSeaParser()
    parser.parse()

    agent_name = parser.args.agent
    agent = DeepSeaExperiment.AGENTS[agent_name]
    n_episodes = parser.args.n_episodes
    k = parser.args.k
    max_steps = parser.args.max_steps

    experiment = DeepSeaExperiment([agent_name, agent], n_episodes, k, max_steps)
    experiment.run(write=True)
