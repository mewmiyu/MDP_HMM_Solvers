import argparse
import os
import time
from typing import Tuple, Union, Any, List, Callable

import numpy as np
from mushroom_rl.core import Agent, Environment, Core
from mushroom_rl.utils.dataset import compute_J


class ExperimentParser:
    """
    Parses the command line arguments for an old.experiment.
    """

    def __init__(self, title: str = None):
        """
        Constructor.

        Args:
            title: The old.experiment title.
        """
        self._parser = argparse.ArgumentParser(description=title)
        self._parser.add_argument('-a', '--agent', metavar='str', type=str, help='The agent to use')
        self._parser.add_argument('-n', '--n_episodes', metavar='int', type=int,
                                  help='he number of episodes to move the agent')
        self._parser.add_argument('-k', '--k', metavar='int', type=int, help='The number of iterations to perform')

        self.args = None

    def parse(self):
        """
        Parses the command line arguments.
        """
        self.args = self._parser.parse_args()


class Experiment:
    """
    Allows to run an old.experiment with a given agent and environment. The result can be saved in a file and plotted as
     one plot later
    """

    DIR_OUTPUT = 'build/'

    def __init__(self, agent: List[Union[str, Callable[..., Agent]]], n_episodes: int, k: int):
        """
        Constructor.

        Args:
            agent: The agent to use
            n_episodes: The number of episodes to move the agent
            k: The number of iterations to perform
        """
        self.agent = agent
        self.n_episodes = n_episodes
        self.k = k

    @staticmethod
    def benchmark(func, *args) -> Tuple[Union[Any, None], float]:
        """
        Benchmarks a function.

        Args:
            func: The function to benchmark.
            *args: The arguments to pass to the function.

        Returns:
            The result of the function and the time it took to execute.
        """
        start = time.time()
        res = func(*args)
        end = time.time()
        return res, end - start

    @staticmethod
    def compute_reward(agent: Agent, env: Environment, n_episodes: int, k: int) -> List[np.ndarray]:
        """
             Computes the reward_k for a given agent (single iteration).

             Args:
                 agent: The agent to evaluate
                 env: The environment to evaluate the agent on
                 n_episodes: The number of episodes to evaluate the agent on
                 k: The number of iterations to perform

             Returns:
                 The reward_k for the given agent.
             """
        reward_k = list()
        for seed in range(k):
            # Set the seed
            np.random.seed(seed)

            # Reinforcement learning old
            core = Core(agent, env)
            # Train
            core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False)
            # Evaluate results for n_episodes
            dataset_q = core.evaluate(n_episodes=1, render=False)
            # Compute the average objective value
            reward = np.mean(compute_J(dataset_q, 1))
            reward_k.append(reward)
        return reward_k

    @staticmethod
    def save(filename: str, result: list):
        """
        Saves the result in a file.

        Args:
            filename: The filename to save the result in.
            result: The result to save
        """
        os.makedirs(Experiment.DIR_OUTPUT, exist_ok=True)
        np.save(f'{Experiment.DIR_OUTPUT}{filename}.npy', result)

    @staticmethod
    def create_labels(*args: str) -> List[str]:
        """
        Creates the labels for the plots.

        Args:
            labels: The labels to use

        Returns:
            The labels for the plots.
        """
        return list(map(lambda l: l.replace(Experiment.DIR_OUTPUT, ''), args))

    @staticmethod
    def plot(filenames: List[str], alphas: List[float], markers: Tuple[str], *args):
        """
        Plots the results.

        Args:
            filenames: The name of the files to read the results from
            alphas: The alpha values used
            markers: The markers used
            *args: Additional arguments
        """
        raise NotImplementedError

    def run(self, write: bool = False) -> List[List[np.ndarray]]:
        """
        Runs the old.

        Args:
            write: If True, the results are written in a file. (agent name.npy)

        Returns:
            A list of lists of rewards.
        """
        raise NotImplementedError
