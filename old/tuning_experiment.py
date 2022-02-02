import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.environments import GridWorld

"""
Source: https://mushroomrl.readthedocs.io/en/latest/source/tutorials/tutorials.0_experiments.html
"""


if __name__ == '__main__':
    from mushroom_rl.core import Core
    from g_learning import GLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.utils.dataset import compute_J

    for size in range(10, 16, 2):
        beta = [0.001, 0.01, 0.1, 1]
        for hyp in range(len(beta)):
            all_j = list()

            for k in range(24):
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

                agent = GLearning(env.info, pi, learning_rate=learning_rate, beta_linear=beta[hyp])
                # Reinforcement learning experiment
                core = Core(agent, env)

                n_episodes = 1
                j = list()

                for i in range(10):
                    # Evaluate results for n_episodes
                    dataset = core.evaluate(n_episodes=n_episodes, render=False)
                    # Compute the average objective value
                    j.append(np.mean(compute_J(dataset, env.info.gamma)))
                    # Train
                    core.learn(n_steps=1000, n_steps_per_fit=1, render=False)
                all_j.append(j)

            all_j = np.array(all_j)
            steps = np.arange(0, 10000, 1000)
            # Compute the 10, 50, 90-th percentiles and plot them
            p10, p50, p90 = np.percentile(all_j, [10, 50, 90], 0)
            plt.fill_between(steps, p10, p90, where=p90 >= p10, label=f'[10-90] percentiles for k={beta[hyp]}', alpha=0.2)
            plt.plot(steps, p50, label=f'median for k={beta[hyp]}')
            plt.xlabel('steps')
            plt.ylabel('cumulative discounted reward')
            plt.title(f'{size}''x'f'{size} Gridworld Experiment: G-Learning')
            plt.legend()
        plt.show()

