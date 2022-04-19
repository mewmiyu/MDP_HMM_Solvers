import numpy as np
from policy_iteration import policy_iteration

"""
    Source: 
    https://github.com/MushroomRL/mushroom-rl/blob/cb7dc7503ad1a8492e4fcf360a23570c3f3e55b8/tests/solvers/test_dynamic_programming.py
"""


def test_policy_iteration():
    p = np.array([[[1., 0., 0., 0.],
                   [0.1, 0., 0.9, 0.],
                   [1., 0., 0., 0.],
                   [0.1, 0.9, 0., 0.]],
                  [[0., 1., 0., 0.],
                   [0., 0.1, 0., 0.9],
                   [0.9, 0.1, 0., 0.],
                   [0., 1., 0., 0.]],
                  [[0.9, 0., 0.1, 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0.1, 0.9]],
                  [[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]]])
    r = np.array([[[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]],
                  [[0., 0., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]],
                  [[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 1.]],
                  [[0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.],
                   [0., 0., 0., 0.]]])
    gamma = .95
    theta = 0.

    q, p = policy_iteration(p, r, gamma, theta)
    q_test = np.array([0.93953176, 0.99447514, 0.99447514, 0.])
    p_test = np.array([1, 1, 3, 0])

    assert np.allclose(q, q_test) and np.allclose(p, p_test)
