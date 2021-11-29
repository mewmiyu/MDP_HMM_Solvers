import numpy as np
from mushroom_rl.core import Agent
from mushroom_rl.utils.table import Table


def _parse(sample):
    """
    Utility to parse the sample.
    Args:
         sample (list): the current episode step.
    Returns:
        A tuple containing state, action, reward, next state and absorbing
    """
    state = sample[0]
    action = sample[1]
    reward = sample[2]
    next_state = sample[3]
    absorbing = sample[4]

    return state, action, reward, next_state, absorbing


class PolicyIteration(Agent):
    def __init__(self, mdp_info, policy, features=None):
        """
        Constructor.
        """
        self.V = Table(mdp_info.observation_space.shape)
        policy.set_v(self.V)

        self._add_save_attr(V='mushroom')
        super().__init__(mdp_info, policy, features)

    def fit(self, dataset):
        theta = 0.001
        delta = 0
        policy_stable = False
        while not policy_stable:
            for sample in dataset:
                print(dataset)
                state, action, reward, next_state, _ = _parse(sample)
                while theta > delta:
                    v_current = self.V[state]
                    self.V[state] = reward + self.mdp_info.gamma * self.V[next_state]
                    delta = np.max([delta, np.linalg.norm(v_current-self.V[state])])
            for sample in dataset:
                state, action, reward, next_state, _ = _parse(sample)
                policy_stable = True
                old_action = self.policy.draw_action(state)
                new_action = np.max(reward + self.mdp_info.gamma * self.V[next_state])
                if not (old_action == new_action):
                    policy_stable = False
                # for now i didnt update the policy cos no idea how

