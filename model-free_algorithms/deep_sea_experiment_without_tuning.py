import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.core import Core
from psi_learning import PsiLearning
from g_learning import GLearning
from deep_sea import DeepSea
from mushroom_rl.algorithms.value.td.q_learning import QLearning
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J


def experiment_deepsea(agent_a):
    e_k = list()
    for k in range(2):
        # Set the seed
        np.random.seed(k)

        # Reinforcement learning experiment
        core = Core(agent_a, env)

        r = 0
        i = 1

        while r < 1 and i < np.power(10, 5):
            # Evaluate results for n_episodes
            dataset_q = core.evaluate(n_episodes=1, render=False)
            # Compute the average objective value
            r += np.mean(compute_J(dataset_q, env.info.gamma)) / i
            # Train
            core.learn(n_episodes=1, n_steps_per_fit=1, render=False)
            i += 1

        e_k.append(i)
    return e_k


if __name__ == '__main__':

    min_size = 6
    max_size = 14
    all_episodes = list()
    all_psi_episodes = list()
    all_g_episodes = list()
    failed_size = max_size
    failed_size_psi = max_size
    failed_size_g = max_size

    for size in range(min_size, max_size, 2):

        # Create the grid environment
        env = DeepSea(size, start=(0, 0), goal=(size-1, size-1))
        # Use an epsilon-greedy policy
        epsilon = Parameter(value=0.1)
        pi = EpsGreedy(epsilon=epsilon)

        learning_rate = Parameter(.1 / 10)

        approximator_params = dict(input_shape=2 * size,
                                   output_shape=(env.info.action_space.n,),
                                   n_actions=env.info.action_space.n)

        a = QLearning(env.info, pi, learning_rate=learning_rate)

        episodes_k = experiment_deepsea(a)
        if np.mean(episodes_k) == np.power(10, 5):
            failed_size = size
        if not failed_size <= size:
            all_episodes.append(np.mean(episodes_k))

        a2 = PsiLearning(env.info, pi, learning_rate=learning_rate)

        episodes_k = experiment_deepsea(a2)

        if np.mean(episodes_k) == np.power(10, 5):
            failed_size_psi = size
        if not failed_size_psi <= size:
            all_psi_episodes.append(np.mean(episodes_k))

        a3 = GLearning(env.info, pi, learning_rate=learning_rate)

        episodes_k = experiment_deepsea(a3)

        if np.mean(episodes_k) == np.power(10, 5):
            failed_size_g = size
        if not failed_size_g <= size:
            all_g_episodes.append(np.mean(episodes_k))

    steps = np.arange(min_size, max_size, 2)
    steps_q = np.arange(min_size, failed_size, 2)
    steps_psi = np.arange(min_size, failed_size_psi, 2)
    steps_g = np.arange(min_size, failed_size_g, 2)
    plt.plot(steps_q, np.array(all_episodes), marker='o', label='q')
    plt.plot(steps_psi, np.array(all_psi_episodes), marker='^', label='psi')
    plt.plot(steps_g, np.array(all_g_episodes), marker='>', label='g')
    plt.plot(steps, [np.power(2, i-1) for i in range(min_size, max_size, 2)], label='2^L-1')
    plt.xlabel('size of gridworld')
    plt.ylabel('number of episodes until cumulative reward is 1')
    plt.title('Gridworld Experiment')
    plt.yscale('symlog', linthresh=0.01)
    plt.legend()
    plt.show()
