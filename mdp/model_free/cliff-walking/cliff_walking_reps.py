from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.algorithms.value.td.q_learning import QLearning
from mushroom_rl.core import Core, Agent, Environment
from mushroom_rl.policy import EpsGreedy, Boltzmann
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from mdp.algo.model_free.env.cliff_walking import CliffWalking
from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.experiment.model_free import Experiment
from mdp.algo.model_free.reps import REPS


def experiment_cliffwalking(agent: Agent, env: Environment, n_episodes: int, k: int) -> List[np.ndarray]:
    r_k = list()
    for k_i in range(k):
        # Set the seed
        np.random.seed(k_i)

        # Reinforcement learning experiment
        core = Core(agent, env)
        # Train
        core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False, quiet=True)
        # Evaluate results for n_episodes
        dataset_q = core.evaluate(n_episodes=1, render=False, quiet=True)
        # Compute the average objective value
        r = np.mean(compute_J(dataset_q, 1))
        r_k.append(r)
        # value_f = list()
        # summed = 0
        # for i in range(len(agent_a.Q[:, 0])):
        # for m in range(len(agent_a.Q[0, :])):
        # summed += agent_a.Q[i, m] * agent_a.policy(i, m)
        # value_f.append(summed)
        # new_value_f = np.zeros((16, 16))
        # counter = 0
        # for q in range(16):
        # for r in range(16):
        # new_value_f[q][r] = value_f[counter]
        # counter += 1
        # print(value_f)
        # heatmap = sns.heatmap(pd.DataFrame(new_value_f), vmin=-8, vmax=7)
        # heatmap.set_title('Value Function Heatmap')
        # plt.show()
    return r_k


def run():
    width = 12
    height = 4
    steps = list()

    k = 10
    n_episodes = 1000

    agents = dict(
        reps = REPS
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

    for p in [0, .1, .15, .2, 0.25]:
        steps.append(p)

        # Create the grid environment
        """
        Basically the parameters are plotted like this:
        0,0 0,1 0,2 
        1,0 1,1 1,2
        2,0 2,1 2,2
        Therefore, to get the agent to start at (2,0), start has to be (2,0)
        """
        env = CliffWalking(width, height, start=(0, 0), goal=(0, width - 1), p=p)

        # Use an epsilon-greedy policy
        epsilon = .1
        pi = Boltzmann(beta=epsilon)

        learning_rate = Parameter(.1 / 10)

        for key, value in agents.items():
            agent = value(env.info, pi, learning_rate=learning_rate)
            reward_k = experiment_cliffwalking(agent, env, n_episodes, k)
            # q_p10, q_p50, q_p90
            q_p = np.percentile(reward_k, q)

            reward_list = rewards[key]
            for r_i, q_pi in zip(reward_list, q_p):
                r_i.append(q_pi)

        sum_reward = 0
        for j in range(width + 1):
            sum_reward -= 1 ** j * (0.05 / width)
        best_reward.append(1 + (0.05 / width) + sum_reward)

    steps = np.array(steps)
    for label, marker, alpha, key in zip(labels, markers, alphas, agents.keys()):
        q_p10, q_p50, q_p90 = rewards[key]
        plt.plot(steps, np.array(q_p50), marker=marker, label=label[0])
        plt.fill_between(steps, q_p10, q_p90, alpha=alpha)

    plt.plot(steps, best_reward, label='best reward')

    plt.xlabel('Probability of choosing a random action')
    plt.ylabel(f'Cumulative average reward after {n_episodes} episodes')
    plt.title(f'Cliff-Walking Experiment for Grid-World of size {width} x {height}')
    plt.yscale('Symlog', linthresh=0.01)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    result, time = Experiment.benchmark(run)
    print(time)
