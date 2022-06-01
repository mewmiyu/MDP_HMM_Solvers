import json
import os
import shutil
from typing import List, Any, Union, Iterable

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class Plotter:
    """
    Allows to plot the results of an experiment.

    Attributes:
        Plotter.DIR_OUTPUT: Directory where the results should be saved
        Plotter.NP_SUFFIX: Numpy file suffix
        Plotter.META_SUFFIX: Metadata file suffix
        Plotter.MARKERS: The available markers for plotting
    """

    DIR_OUTPUT = 'build/'
    NP_SUFFIX = '.npy'
    META_SUFFIX = '_meta.json'
    MARKERS = ['o', '^', '>', '<', 'v']

    def __init__(self, path: str = None):
        """
        Constructor.

        Args:
            path: The directory to the file to plot.
        """
        if path is None:
            # Use default path directory with suffix plot
            i = 0
            new_path = f'{self.DIR_OUTPUT}/plot'
            while os.path.isdir(f'{new_path}{i}'):
                i += 1
            self.path = f'{new_path}{i}/'
        elif path != '':
            # Use default path directory without suffix
            self.path = f'{self.DIR_OUTPUT}{path}/'
        else:
            # Customized path directory
            self.path = f'{self.DIR_OUTPUT}{path}'

    @staticmethod
    def clear():
        """
        Clears the output directory.
        """
        shutil.rmtree(Plotter.DIR_OUTPUT)

    def filenames(self) -> List[str]:
        """
        Returns all filenames in the output directory.

        Returns:
            A list of all filenames in the output directory.
        """
        filenames = []
        print(self.path)
        skip = len(self.NP_SUFFIX)
        for file in os.listdir(self.path):
            filename = os.path.join(self.path, file)
            if os.path.isfile(filename) and filename.endswith(self.NP_SUFFIX):
                filenames.append(filename[:len(filename) - skip])
        return filenames

    @staticmethod
    def load_result(filename: str) -> Union[ndarray, Iterable, int, float, tuple, dict]:
        """
        Load the result of a numpy file.

        Args:
            filename: The file representing a numpy array.

        Returns:
            Returns the read numpy array from a file
        """
        return np.load(filename)

    @staticmethod
    def load_meta(filename: str) -> dict:
        """
        Load the metadata from a file.

        Args:
            filename: The file a json object

        Returns:
            Returns the read metadata as dict
        """
        with open(filename) as json_file:
            return json.load(json_file)

    def save(self, result: List[List[np.ndarray]], filename: str, **kwargs: Any):
        """
        Saves the result in a file.

        Args:
            result: The result to save
            filename: The name of the file in which it should be saved
            kwargs: Additional metadata that should be stored
        """
        os.makedirs(self.path, exist_ok=True)
        path = f'{self.path}{filename}'

        # Numpy file
        np.save(f'{path}{self.NP_SUFFIX}', result)

        # Meta data
        with open(f'{path}{self.META_SUFFIX}', 'w') as file:
            json.dump(kwargs, file, indent=4)

    def plot(self, filenames: List[str], **kwargs: Any):
        """
        Plots the results.

         Args:
            filenames: The filenames to plot
            **kwargs: The arguments used to plot the results
        """
        raise NotImplementedError

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
        alphas: List[float] = kwargs['alphas']
        steps: list = kwargs['steps']
        best_reward: list = kwargs['best_reward']

        steps = np.array(steps)

        alphas = alphas[:len(filenames)]
        markers: List[str] = self.MARKERS[:len(filenames)]
        suffix_labels = ['median', '10:90']
        for filename, alpha, marker in zip(filenames, alphas, markers):
            q_p10, q_p50, q_p90 = self.load_result(filename + self.NP_SUFFIX)
            metadata = self.load_meta(filename + self.META_SUFFIX)
            agent = metadata['agent_short']

            labels = list(map(lambda x: f'{agent}_{x}', suffix_labels))
            plt.plot(steps, q_p50, marker=marker, label=labels[0])
            plt.fill_between(steps, q_p10, q_p90, label=labels[1], alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')
        plt.legend()


class AgentsDeepSeaPlotter(AgentsPlotter):
    """
    Allows to plot a sequence of agents using the experiment Deep Sea.
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
    Allows to plot a sequence of agents using the experiment Cliff Walking.
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
    Allows to plot a sequence of betas of an agent.
    """

    def __init__(self, path: str = None):
        """
        Constructor.

        Args:
            path: The path to the file to plot.
        """
        super().__init__(path)

    def plot(self, filenames: List[str], **kwargs: Any):
        alphas: List[float] = kwargs['alphas']
        steps: list = kwargs['steps']
        best_reward: list = kwargs['best_reward']

        steps = np.array(steps)

        alphas = alphas[:len(filenames)]
        markers: List[str] = self.MARKERS[:len(filenames)]
        for filename, marker, alpha in zip(filenames, markers, alphas):
            q_p10, q_p50, q_p90 = self.load_result(filename + self.NP_SUFFIX)
            metadata = self.load_meta(filename + self.META_SUFFIX)

            beta = str(metadata['beta'])
            plt.plot(steps, q_p50, marker=marker, label='Beta: {}'.format(beta))
            plt.fill_between(steps, q_p10, q_p90, alpha=alpha)

        plt.plot(steps, best_reward, label='Best reward')
        plt.legend()


class BetasDeepSeaPlotter(BetasPlotter):
    """
    Allows to plot a sequence of betas of an agent using the experiment Deep Sea.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        plt.xlabel('Size of gridworld')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        agent: str = self.load_meta(filenames[0]+ self.META_SUFFIX)['agent_full']
        plt.title(f'Deep Sea Experiment - {agent}')
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
    Cliff Walking experiment_oldbeta plotter.
    """

    def plot(self, filenames: List[str], **kwargs: Any):
        n_episodes: int = kwargs['n_episodes']
        width: int = kwargs['width']
        height: int = kwargs['height']
        plt.xlabel('Probability of choosing a random action')
        plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
        agent: str = self.load_meta(filenames[0]+ self.META_SUFFIX)['agent_full']
        plt.title(f'Cliff-Walking Experiment for Grid-World of size {width} x {height} - {agent}')
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
