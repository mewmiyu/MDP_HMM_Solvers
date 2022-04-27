import time
from typing import List, Tuple, Union, Any, Callable, Dict

import numpy as np
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Environment, Agent
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import to_parameter

from mdp.algo.model_free.env.cliff_walking import CliffWalking
from mdp.algo.model_free.env.deep_sea import DeepSea
from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.reps import REPS


class Experiment:
    """
    Allows to run an experiment with a given agent and environment. The result can be saved in a file and plotted as
    one plot later.

    Attributes:
        Experiment.AGENTS: A list of all agents
        Experiment.AGENTS: A list of all agents name
        Experiment.Q: A list of q values for the percentile. The values must be between 0 and 100 inclusive
    """

    AGENTS: Dict[str, Callable[..., Agent]] = dict(
        q=QLearning,
        psi=PsiLearning,
        g=GLearning,
        mirl=MIRL,
        reps=REPS
    )

    AGENTS_NAME: Dict[str, str] = dict(
        q='Q Learning',
        psi='Psi Learning',
        g='G Learning',
        mirl='MIRL',
        reps='REPS'
    )

    Q: List[int] = [10, 50, 90]

    def __init__(self, agent_constructor: Callable[..., Agent], **kwargs: int):
        """
        Constructor.

        Args:
            agent_constructor: The agent to use
            **kwargs: Arguments that will be used when running the experiment
        """
        self.agent_constructor = agent_constructor
        self.n_episodes = kwargs.get('n_episodes', 100)
        self.k = kwargs.get('k', 10)
        self.alpha = to_parameter(kwargs.get('alpha', .1))
        self.epsilon = to_parameter(kwargs.get('epsilon', .01))

    def run(self) -> List[List[np.ndarray]]:
        """
        Runs the experiment.

        Returns:
            A list of lists of rewards.
        """
        raise NotImplementedError

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
            try:
                core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False)
            except Exception:
                core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False)
            # Evaluate results for n_episodes
            dataset_q = core.evaluate(n_episodes=1, render=False)
            # Compute the average objective value
            reward = np.mean(compute_J(dataset_q, 1))
            reward_k.append(reward)
        return reward_k


class DeepSeaExperiment(Experiment):

    def __init__(self, agent_constructor: Callable[..., Agent], **kwargs: int):
        """
        Constructor.

        Args:
            agent_constructor: The agent to use
            **kwargs: Arguments that will be used when running the experiment
        """
        super().__init__(agent_constructor, **kwargs)
        self.max_steps = kwargs.get('max_steps', 10)

    def run(self) -> List[List[np.ndarray]]:
        # Prepare result list
        q_ps: List[List[np.ndarray]] = list()
        for _ in self.Q:
            q_ps.append(list())

        # Run the experiment
        for exponent in range(1, self.max_steps + 1):
            size = np.power(2, exponent)

            # Use an epsilon-greedy policy
            pi = EpsGreedy(epsilon=self.epsilon)

            # Create the environment
            env = DeepSea(size, start=(0, 0), goal=(size - 1, size - 1))
            agent = self.agent_constructor(env.info, pi, self.alpha)

            # q_p10, q_p50, q_p90
            reward_k = self.compute_reward(agent, env, self.n_episodes, self.k)

            for q_pi, q_p in zip(q_ps, np.percentile(reward_k, self.Q)):
                q_pi.append(q_p)
        return q_ps


class CliffWalkingExperiment(Experiment):

    def __init__(self, agent_constructor: Callable[..., Agent], **kwargs: int):
        """
        Constructor.

        Args:
            agent_constructor: The agent to use
            **kwargs: Arguments that will be used when running the experiment
        """
        super().__init__(agent_constructor, **kwargs)
        self.width = kwargs.get('width', 12)
        self.height = kwargs.get('height', 4)
        self.p = kwargs.get('p', [0, .1, .2])

    def run(self) -> List[List[np.ndarray]]:
        # Prepare result list
        q_ps: List[List[np.ndarray]] = list()
        for _ in self.Q:
            q_ps.append(list())

        # Run the experiment
        for p_i in self.p:
            # Use an epsilon-greedy policy
            pi = EpsGreedy(epsilon=self.epsilon)

            # Create the grid environment
            """
            Basically the parameters are plotted like this:
            0,0 0,1 0,2 
            1,0 1,1 1,2
            2,0 2,1 2,2
            Therefore, to get the agent to start at (2,0), start has to be (2,0)
            """
            env = CliffWalking(self.width, self.height, start=(0, 0), goal=(0, self.width - 1), p=p_i)
            agent = self.agent_constructor(env.info, pi, self.alpha)

            # q_p10, q_p50, q_p90
            reward_k = self.compute_reward(agent, env, self.n_episodes, self.k)

            for q_pi, q_p in zip(q_ps, np.percentile(reward_k, self.Q)):
                q_pi.append(q_p)
        return q_ps
