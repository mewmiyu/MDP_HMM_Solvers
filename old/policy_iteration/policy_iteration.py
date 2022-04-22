from copy import deepcopy

import numpy as np


def policy_iteration(transition, reward, gamma, theta):
    """
    Computes the optimal value function and policy by using policy iteration

    Args:
        transition: matrix of transition probabilities
        reward: vector of rewards
        gamma: discount factor
        theta: small positive number determining6:the accuracy of estimation

    Returns:
        The optimal value function and policy
    """
    n_s, n_a, _ = transition.shape

    pi = np.zeros(n_s, dtype=int)
    value_function = np.zeros(n_s)
    policy_stable = False

    while not policy_stable:
        while True:
            # copy value function for the value iteration
            old_value_function = deepcopy(value_function)
            delta = 0
            # need to recreate this for theta instead of eps & change the loop
            for state in range(n_s):
                action = pi[state]
                # compute p(s' |s, pi(a))
                t = transition[state, action, :]
                # compute reward depending on state and pi(a)
                r = reward[state, action, :]
                # compute value of state
                value_function[state] = t.dot(r + gamma * old_value_function)
                # compute, whether there is a significant change between the old value and new value of state
                delta = np.max([delta, np.linalg.norm(value_function[state] - old_value_function[state])])
            if delta <= theta:
                break
        # policy evaluation complete
        policy_stable = True
        for state in range(n_s):
            # current value of the state, taken as maximum value, then updated throughout the algorithm
            v_max = value_function[state]
            for action in range(n_a):
                # compute p(s' |s, a)
                t = transition[state, action, :]
                # compute reward depending on state and action
                r = reward[state, action, :]
                # compute value of state for this action
                v_a = t.T.dot(r + gamma * value_function)
                # if value of state with the current action is higher than with original policy, update policy
                if v_a > v_max:
                    pi[state] = action
                    v_max = v_a
                    # we updated the policy, so we need to do policy evaluation again
                    policy_stable = False
    return value_function, pi
