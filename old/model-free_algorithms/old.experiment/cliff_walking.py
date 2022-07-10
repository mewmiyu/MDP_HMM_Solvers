from typing import Dict, List, Union, Callable, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Agent
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter

from mdp.algo.model_free.env.cliff_walking import CliffWalking
from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.psi_kl import REPS
from mdp.experiment.model_free import ExperimentParser, Experiment


class CliffWalkingParser(ExperimentParser):
    """
    Parses the command line arguments for a Cliff Walking old.experiment.
    """

    def __init__(self):
        super().__init__('Cliff Walking')
        self._parser.add_argument('-w', '--width', metavar='int', type=int, help='Map width')
        self._parser.add_argument('-he', '--height', metavar='int', type=int, help='Map height')
        self._parser.add_argument('-p', '--probabilities', type=float, nargs="+",
                                  help='Probabilities')


class CliffWalkingExperiment(Experiment):
    """
    The Cliff Walking old.experiment.

    Parameters:
        CliffWalkingExperiment.AGENTS: A list of available agents
        CliffWalkingExperiment.Q: A list of q values for the percentile. The values must be between 0 and 100 inclusive

    """
    AGENTS: Dict[str, Agent] = dict(
        q=QLearning,
        psi=PsiLearning,
        g=GLearning,
        mirl=MIRL,
        reps=REPS
    )

    Q: List[int] = [10, 50, 90]

    def __init__(self, agent: List[Union[str, Callable[..., Agent]]], n_episodes: int, k: int, width: int, height: int,
                 p: List[float]):
        """
        Constructor.

        Args:
            agent: The agent to use
            n_episodes: The number of episodes to move the agent
            k: The number of iterations to perform
            width: The width of the grid-world
            height: The height of the grid-world
            p: The list of probabilities to use
        """
        super().__init__(agent, n_episodes, k)
        self.width = width
        self.height = height
        self.p = p

    @staticmethod
    def plot(filenames: List[str], alphas: List[float], markers: Tuple[str], *args):
        width, height, n_episodes, p = args
        experiment_name = CliffWalkingExperiment.__name__

        steps = [p_i for p_i in p]
        best_reward = list()
        for _ in p:
            sum_reward = 0
            for j in range(width + 1):
                sum_reward -= 1 ** j * (0.5 / width)
            best_reward.append(10 + (0.5 / width) + sum_reward)

        steps = np.array(steps)
        for filename, alpha, marker in zip(filenames, alphas, markers):
            agent_name = filename.replace(f'{experiment_name}_', '').replace('.npy', '')
            q_p10, q_p50, q_p90 = np.load(filename)

            labels = Experiment.create_labels(f'{agent_name}_median', f'{agent_name}_10:90')
            plt.plot(steps, q_p50, marker=marker, label=labels[0])
            plt.fill_between(steps, q_p10, q_p90, label=labels[1], alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')

        plt.xlabel('Probability of choosing a random action')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        plt.title(f'Cliff-Walking Experiment for Grid-World of size {width} x {height}')
        plt.yscale('Symlog', linthresh=0.01)
        plt.legend()
        plt.show()

    def run(self, write: bool = False) -> List[List[np.ndarray]]:
        q_ps: List[List[np.ndarray]] = list()
        for _ in self.Q:
            q_ps.append(list())

        for p_i in self.p:

            # Use an epsilon-greedy policy
            epsilon = .1
            pi = EpsGreedy(epsilon=epsilon)

            # Create the grid environment
            """
            Basically the parameters are plotted like this:
            0,0 0,1 0,2 
            1,0 1,1 1,2
            2,0 2,1 2,2
            Therefore, to get the agent to start at (2,0), start has to be (2,0)
            """
            env = CliffWalking(self.width, self.height, start=(0, 0), goal=(0, self.width - 1), p=p_i)
            learning_rate = Parameter(.1 / 10)
            agent = self.agent[1](env.info, pi, learning_rate)

            # q_p10, q_p50, q_p90
            reward_k = self.compute_reward(agent, env, self.n_episodes, self.k)

            for q_pi, q_p in zip(q_ps, np.percentile(reward_k, self.Q)):
                q_pi.append(q_p)

            if write:
                self.save(f'{CliffWalkingExperiment.__name__}_{self.agent[0]}', q_ps)
        return q_ps


if __name__ == '__main__':
    parser = CliffWalkingParser()
    parser.parse()

    agent_name = parser.args.agent
    agent = CliffWalkingExperiment.AGENTS[agent_name]
    n_episodes = parser.args.n_episodes
    k = parser.args.k
    width = parser.args.width
    height = parser.args.height
    p = parser.args.probabilities

    experiment = CliffWalkingExperiment([agent_name, agent], n_episodes, k, width, height, p)
    experiment.run(write=True)
