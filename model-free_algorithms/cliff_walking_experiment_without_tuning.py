import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.core import Core
from psi_learning import PsiLearning
from g_learning import GLearning
from mirl import MIRL
from cliff_walking import CliffWalking
from mushroom_rl.algorithms.value.td.q_learning import QLearning
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J


def experiment_cliffwalking(agent_a):
    r_k = list()
    for k in range(10):
        # Set the seed
        np.random.seed(k)

        # Reinforcement learning experiment
        core = Core(agent_a, env)
        # Train
        core.learn(n_episodes=100, n_steps_per_fit=1, render=False)
        # Evaluate results for n_episodes
        dataset_q = core.evaluate(n_episodes=1, render=False)
        # Compute the average objective value
        r = np.mean(compute_J(dataset_q, 1))
        r_k.append(r)

    return r_k


if __name__ == '__main__':

    min_size = 2
    size = min_size
    counter = 2
    max_size = 17

    steps = list()
    all_reward = list()
    all_reward_p10 = list()
    all_reward_p90 = list()

    all_psi_reward = list()
    all_psi_reward_p10 = list()
    all_psi_reward_p90 = list()

    all_g_reward = list()
    all_g_reward_p10 = list()
    all_g_reward_p90 = list()

    all_mirl_reward = list()
    all_mirl_reward_p10 = list()
    all_mirl_reward_p90 = list()

    best_reward = list()

    while size < max_size:
        steps.append(size)

        # Create the grid environment
        env = CliffWalking(size, start=(0, 0), goal=(size-1, size-1))
        # Use an epsilon-greedy policy
        epsilon = Parameter(value=0.1)
        pi = EpsGreedy(epsilon=epsilon)

        learning_rate = Parameter(.1 / 10)

        approximator_params = dict(input_shape=2 * size,
                                   output_shape=(env.info.action_space.n,),
                                   n_actions=env.info.action_space.n)

        a = QLearning(env.info, pi, learning_rate=learning_rate)

        reward_k = experiment_cliffwalking(a)
        q_p10, q_p50, q_p90 = np.percentile(reward_k, [10, 50, 90])
        all_reward.append(q_p50)
        all_reward_p10.append(q_p10)
        all_reward_p90.append(q_p90)

        a2 = PsiLearning(env.info, pi, learning_rate=learning_rate)

        reward_k = experiment_cliffwalking(a2)
        psi_p10, psi_p50, psi_p90 = np.percentile(reward_k, [10, 50, 90])
        all_psi_reward.append(psi_p50)
        all_psi_reward_p10.append(psi_p10)
        all_psi_reward_p90.append(psi_p90)

        a3 = GLearning(env.info, pi, learning_rate=learning_rate)

        reward_k = experiment_cliffwalking(a3)
        g_p10, g_p50, g_p90 = np.percentile(reward_k, [10, 50, 90])
        all_g_reward.append(g_p50)
        all_g_reward_p10.append(g_p10)
        all_g_reward_p90.append(g_p90)

        a4 = MIRL(env.info, pi, learning_rate=learning_rate)

        reward_k = experiment_cliffwalking(a4)
        mirl_p10, mirl_p50, mirl_p90 = np.percentile(reward_k, [10, 50, 90])
        all_mirl_reward.append(mirl_p50)
        all_mirl_reward_p10.append(mirl_p10)
        all_mirl_reward_p90.append(mirl_p90)

        sum_reward = 0
        for j in range(size-2):
            sum_reward -= 1 ** j * (0.01 / size)
        best_reward.append(1 + (0.01 / size) + sum_reward)

        size = np.power(2, counter)
        counter += 1

    steps = np.array(steps)
    plt.plot(steps, np.array(all_reward), marker='o', label='q_median')
    plt.fill_between(steps, all_reward_p10, all_reward_p90, label='q_10:90', alpha=0.3)

    plt.plot(steps, np.array(all_psi_reward), marker='^', label='psi_median')
    plt.fill_between(steps,  all_psi_reward_p10, all_psi_reward_p90, label='psi_10:90', alpha=0.25)

    plt.plot(steps, np.array(all_g_reward), marker='>', label='g_median')
    plt.fill_between(steps, all_g_reward_p10, all_g_reward_p90, label='g_10:90', alpha=0.2)

    plt.plot(steps, np.array(all_mirl_reward), marker='<', label='mirl_median')
    plt.fill_between(steps, all_mirl_reward_p10, all_mirl_reward_p90, label='mirl_10:90', alpha=0.15)

    plt.plot(steps, best_reward, label='best reward')

    plt.xlabel('size of gridworld')
    plt.ylabel('cumulative average reward after 1000 episodes')
    plt.title('Cliff-Walking Experiment')
    plt.legend()
    plt.show()
