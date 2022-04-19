import time
from concurrent.futures import Future
from typing import List, Tuple, Any, Union

import numpy as np
from mushroom_rl.core import Agent, Environment, Core
from mushroom_rl.utils.dataset import compute_J


class Experiment:
    """
    Allows to run an experiment concurrently with a set of agents and an environment parallel.
    """

    def __init__(self, n_episodes: int, k: int, max_workers: int = None, quiet: bool = True):
        """
        Constructor.

        Args:
            n_episodes: The number of episodes to move the agent
            k: The number of iterations to perform
            max_workers: The maximum number of workers to use
            quiet: If True, the experiment will not print progress bar
        """
        self.n_episodes = n_episodes
        self.k = k
        self.max_workers = max_workers
        self.quiet = quiet

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
    def collect(futures: List[Future]) -> list:
        """
        Collects the results of the futures.

        Args:
            futures: The list of future results to collect.

        Returns:
            The collected results.

        Raises:
            Exception: If the call raised an exception. That exception will be raised.
        """
        results = list()
        for future in futures:
            results.append(future.result())
        return results

    def is_serial(self) -> bool:
        """
        Returns True if the experiment will serial executed, False otherwise.
        """
        return self.max_workers == 1

    def _compute_reward(self, agent: Agent, env: Environment) -> np.ndarray:
        """
        Computes the reward_k for a given agent (single iteration).

        Args:
            agent: The agent to evaluate
            env: The environment to evaluate the agent on

        Returns:
            The reward_k for the given agent.
        """
        # Reinforcement learning experiment
        core = Core(agent, env)
        # Train
        core.learn(n_episodes=self.n_episodes, n_steps_per_fit=1, render=False, quiet=self.quiet)
        # Evaluate results for n_episodes
        dataset_q = core.evaluate(n_episodes=1, render=False, quiet=self.quiet)
        # Compute the average objective value
        return np.mean(compute_J(dataset_q, 1))

    def compute_reward(self, agent: Agent, env: Environment) -> List[np.ndarray]:
        """
        Computes the reward_k for a given (single) agent.

        Args:
            agent: The agent to evaluate
            env: The environment to evaluate the agent on

        Returns:
            The list of reward_k for the given agent.
        """
        reward_k = list()
        for k in range(self.k):
            # Set the seed
            np.random.seed(k)
            reward_k.append(self._compute_reward(agent, env))
        return reward_k

    def run(self):
        """
        Runs the experiment.
        """
        raise NotImplementedError
