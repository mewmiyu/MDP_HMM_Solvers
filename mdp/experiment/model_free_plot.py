import os
import shutil
from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt

from mdp.experiment.model_free import Experiment


class Plotter:
    """
    Allows to plot the results of an experiment.

    Attributes:
        Plotter.DIR_OUTPUT: Directory where the results should be saved
    """

    DIR_OUTPUT = 'build/'

    def __init__(self, path: str = None):
        """
        Constructor.

        Args:
            path: The path to the file to plot.
        """
        if path is None:
            i = 0
            new_path = f'{self.DIR_OUTPUT}/plot'
            while os.path.isdir(f'{new_path}{i}'):
                i += 1
            self.path = f'{new_path}{i}'
        elif path != '':
            self.path = f'{self.DIR_OUTPUT}{path}/'
        else:
            self.path = f'{self.DIR_OUTPUT}{path}'

    @staticmethod
    def clear():
        """
        Clears the output directory.
        """
        shutil.rmtree(Plotter.DIR_OUTPUT)

    def save(self, filename: str, result: List[List[np.ndarray]]):
        """
        Saves the result in a file.

        Args:
            filename: The filename to save the result in
            result: The result to save
        """
        os.makedirs(self.path, exist_ok=True)
        np.save(f'{self.path}{filename}.npy', result)

    def plot(self, filenames: List[str], **kwargs: Any):
        """
        Plots the results.

        Args:
            filenames: The filenames to plot
            **kwargs: The arguments used to plot the results
        """
        raise NotImplementedError

    def filenames(self) -> List[str]:
        """
        Returns all filenames in the output directory.

        Returns:
            A list of all filenames in the output directory.
        """
        filenames = []
        for file in os.listdir(self.path):
            filename = os.path.join(self.path, file)
            if os.path.isfile(filename):
                filenames.append(filename)
        return filenames

    def agent_name(self, filename: str) -> str:
        """
        Returns the name of the agent.

        Args:
            filename: The filename to get the name of the agent from

        Returns:
            The name of the agent
        """
        return filename.split('_')[-1].split('.')[0]

    def show(self):
        """
        Shows the plots.
        """
        plt.show()


class AgentsPlotter(Plotter):
    """
    Allows to plot a sequence of agents.
    """

    def __init__(self, path: str = None):
        """
        Constructor.

        Args:
            path: The path to the file to plot.
        """
        super().__init__(path)

    def plot(self, filenames: List[str], **kwargs: Any):
        alphas: List[float] = kwargs['plot_args']['alphas']
        markers: List[str] = kwargs['plot_args']['markers']
        steps: list = kwargs['steps']
        best_reward: list = kwargs['best_reward']

        steps = np.array(steps)

        for filename, alpha, marker in zip(filenames, alphas, markers):
            q_p10, q_p50, q_p90 = np.load(filename)
            agent_name = self.agent_name(filename)
            labels = list(map(lambda x: f'{agent_name}_{x}',
                              ['median', '10:90']))
            plt.plot(steps, q_p50, marker=marker, label=labels[0])
            plt.fill_between(steps, q_p10, q_p90, label=labels[1], alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')
        plt.legend()


class AgentsDeepSeaPlotter(AgentsPlotter):
    """
    Deep Sea experiment plotter.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        plt.xlabel('Size of gridworld')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        plt.title('Deep Sea Experiment')
        plt.tight_layout()
        plt.grid(True)

        best_reward = list()
        steps = list()
        for exponent in kwargs['steps']:
            size = np.power(2, exponent)
            steps.append(size)

            sum_reward = 0
            for j in range(size - 2):
                sum_reward -= 1 ** j * (0.01 / size)
            best_reward.append(1 + (0.01 / size) + sum_reward)

        kwargs['best_reward'] = best_reward
        kwargs['steps'] = steps
        super().plot(filenames, **kwargs)


class AgentsCliffWalkingPlotter(AgentsPlotter):
    """
    Cliff Walking experiment plotter.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        width: int = kwargs['width']
        height: int = kwargs['height']
        plt.xlabel('Probability of choosing a random action')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        plt.title(f'Cliff-Walking Experiment for Grid-World of size {width} x {height}')
        plt.yscale('Symlog', linthresh=0.01)
        plt.tight_layout()
        plt.grid(True)

        best_reward = list()
        for _ in kwargs['steps']:
            sum_reward = 0
            for j in range(width + 1):
                sum_reward -= 1 ** j * (0.5 / width)
            best_reward.append(10 + (0.5 / width) + sum_reward)
        kwargs['best_reward'] = best_reward
        super().plot(filenames, **kwargs)


class BetasPlotter(Plotter):
    """
    Allows to plot a sequence of alphas
    """

    def __init__(self, path: str = None):
        """
        Constructor.

        Args:
            path: The path to the file to plot.
        """
        super().__init__(path)

    def plot(self, filenames: List[str], **kwargs: Any):
        alphas: List[float] = kwargs['plot_args']['alphas']
        markers: List[str] = kwargs['plot_args']['markers']
        betas: List[float] = kwargs['plot_args']['betas']
        steps: list = kwargs['steps']
        best_reward: list = kwargs['best_reward']

        for filename, marker, beta, alpha in zip(filenames, markers, betas, alphas):
            q_p10, q_p50, q_p90 = np.load(filename)
            plt.plot(steps, q_p50, marker=marker, label='Betas: {}'.format(beta))
            plt.fill_between(steps, q_p10, q_p90, alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')
        plt.legend()


class BetasDeepSeaPlotter(BetasPlotter):
    """
    Deep Sea experiment plotter.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        plt.xlabel('Size of gridworld')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        agent_name: str = Experiment.AGENTS_NAME[self.agent_name(filenames[0])]
        plt.title(f'Deep Sea Experiment - {agent_name}')
        plt.tight_layout()
        plt.grid(True)

        best_reward = list()
        steps = list()
        for exponent in kwargs['steps']:
            size = np.power(2, exponent)
            steps.append(size)

            sum_reward = 0
            for j in range(size - 2):
                sum_reward -= 1 ** j * (0.01 / size)
            best_reward.append(1 + (0.01 / size) + sum_reward)

        kwargs['best_reward'] = best_reward
        kwargs['steps'] = steps
        super().plot(filenames, **kwargs)


class BetasCliffWalkingPlotter(BetasPlotter):
    """
    Cliff Walking experiment plotter.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        width: int = kwargs['width']
        height: int = kwargs['height']
        plt.xlabel('Probability of choosing a random action')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        agent_name: str = Experiment.AGENTS_NAME[self.agent_name(filenames[0])]
        plt.title(f'Cliff-Walking Experiment for Grid-World of size {width} x {height} - {agent_name}')
        plt.yscale('Symlog', linthresh=0.01)
        plt.tight_layout()
        plt.grid(True)

        best_reward = list()
        for _ in kwargs['steps']:
            sum_reward = 0
            for j in range(width + 1):
                sum_reward -= 1 ** j * (0.5 / width)
            best_reward.append(10 + (0.5 / width) + sum_reward)
        kwargs['best_reward'] = best_reward
        super().plot(filenames, **kwargs)
