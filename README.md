# Reinforcement Learning Algorithms with Inference

In this project the reinforcement learning algorithms with inference
are compared to their counterparts without inference.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
In the folder mdp.algo.bandit are the experiments for the bandit algorithm,
the relative payoff procedure and the comparison of both, respectively.
In the folder mdp.model-free are the experiments for the cliff-walking
and deep sea environments. There are experiments for the tuning of the
inverse temperature beta for each
model-free algorithm with inference (G-learning, Psi-learning and MIRL) and
the experiment for the comparison of the algorithms to Q-learning. Additionally,
there are two experiments (deep sea and cliff-walking) for the algorithm called
Psi-Auto, an algorithm based on Psi-learning that tries to automatically tune
the inverse temperature by minimizing a dual function. This dual function is dependent on the advantage
over the state action space and the initial value for the inverse temperature.

## Technologies
The Project is created with:
* Python: 3.8
* numpy: 1.17.0
* mushroom-rl: 1.7.0
* matplotlib: 3.4.3
* scipy: 1.7.1
* pandas: 1.4.1
* seaborn: 0.11.2
	
## Setup

This project uses the Mushroom-RL library.
For setup and installation of Mushroom-RL refer to
```
https://github.com/MushroomRL/mushroom-rl
```

Additionally, to run this project, install the requirements as specified
in requirements.txt. For that you will need PyTorch and Anaconda/Miniconda.

With IDE, such as Pycharm, you can create a conda environment by
using the tutorial
```
https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#569eec8d
```

Then you can install pytorch with
```
 conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Then you can execute the experiments
manually, if you go to the respective folder as described in
(#general-info).




