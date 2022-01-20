import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.environments import GridWorld

if __name__ == '__main__':
    from mushroom_rl.core import Core
    from psi_learning import PsiLearning
    from mushroom_rl.algorithms.value.td.q_learning import QLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.utils.dataset import compute_J

    for size in range(10, 20, 2):
        all_j_q = list()
        all_j_psi = list()

        for k in range(50):
            # Set the seed
            np.random.seed(k)

            # Create the grid environment
            env = GridWorld(height=size, width=size, start=(0, 0), goal=(5, 5))
            # Use an epsilon-greedy policy
            epsilon = Parameter(value=0.1)
            pi = EpsGreedy(epsilon=epsilon)

            env.reset()

            learning_rate = Parameter(.1 / 10)

            approximator_params = dict(input_shape=2*size,
                                       output_shape=(env.info.action_space.n,),
                                       n_actions=env.info.action_space.n)

            agent = QLearning(env.info, pi, learning_rate=learning_rate)
            agent2 = PsiLearning(env.info, pi, learning_rate=learning_rate)
            # Reinforcement learning experiment
            core = Core(agent, env)
            core2 = Core(agent2, env)

            n_episodes = 1
            j_q = list()
            j_psi = list()

            for i in range(20):
                # Evaluate results for n_episodes
                dataset_q = core.evaluate(n_episodes=n_episodes, render=False)
                dataset_psi = core2.evaluate(n_episodes=n_episodes, render=False)
                # Compute the average objective value
                j_q.append(np.mean(compute_J(dataset_q, env.info.gamma)))
                j_psi.append(np.mean(compute_J(dataset_psi, env.info.gamma)))
                # Train
                core.learn(n_steps=1000, n_steps_per_fit=1, render=False)
                core2.learn(n_steps=1000, n_steps_per_fit=1, render=False)
            all_j_q.append(j_q)
            all_j_psi.append(j_psi)

        all_j_q = np.array(all_j_q)
        all_j_psi = np.array(all_j_psi)
        steps = np.arange(0, 20000, 1000)
        # Compute the 10, 50, 90-th percentiles and plot them
        q_p10, q_p50, q_p90 = np.percentile(all_j_q, [10, 50, 90], 0)
        psi_p10, psi_p50, psi_p90 = np.percentile(all_j_psi, [10, 50, 90], 0)
        plt.fill_between(steps, q_p10, q_p90, where=q_p90 >= q_p10, label='q: [10-90] percentiles', alpha=0.2)
        plt.plot(steps, q_p50, label='q: median')
        plt.fill_between(steps, psi_p10, psi_p90, where=psi_p90 >= psi_p10, label='psi: [10-90] percentiles',
                         alpha=0.2)
        plt.plot(steps, psi_p50, label='psi: median')
        plt.xlabel('steps')
        plt.ylabel('cumulative discounted reward')
        plt.title(f'{size}''x'f'{size} Gridworld Experiment: Q-Learning vs Psi-Learning')
        plt.legend()
        plt.show()