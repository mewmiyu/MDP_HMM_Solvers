import numpy as np
from mushroom_rl.environments import GridWorld

"""
Source: Mushroom Tutorial
"""

if __name__ == '__main__':
    from mushroom_rl.core import Core
    from psi_learning import PsiLearning
    from mushroom_rl.policy import EpsGreedy
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.utils.dataset import compute_J

    # Set the seed
    np.random.seed(1)

    # Create the grid environment
    env = GridWorld(height=5, width=5, start=(0, 0), goal=(2, 2))
    # Using an epsilon-greedy policy
    epsilon = Parameter(value=0.1)
    pi = EpsGreedy(epsilon=epsilon)

    env.reset()
    env.render()

    learning_rate = Parameter(.1 / 10)

    approximator_params = dict(input_shape=10,
                               output_shape=(env.info.action_space.n,),
                               n_actions=env.info.action_space.n)

    agent = PsiLearning(env.info, pi, learning_rate=learning_rate)
    print(env.info)
    # Reinforcement learning experiment
    core = Core(agent, env)

    # Visualize initial policy for 3 episodes
    dataset = core.evaluate(n_episodes=3, render=False)

    # Print the average objective value before learning
    J = np.mean(compute_J(dataset, env.info.gamma))
    print(f'Objective function before learning: {J}')

    # Train
    core.learn(n_steps=10, n_steps_per_fit=1, render=False)

    # Visualize results for 3 episodes
    dataset = core.evaluate(n_episodes=3, render=False)

    # Print the average objective value after learning
    print(compute_J(dataset, env.info.gamma))
    J = np.mean(compute_J(dataset, env.info.gamma))
    print(f'Objective function after learning: {J}')