from typing import List

from mushroom_rl.algorithms.value import QLearning

from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.psi_kl import REPS
from mdp.experiment import CliffWalkingExperiment
from mdp.experiment import DeepSeaExperiment
from mdp.experiment.model_free import Experiment


def _select_entry(entries: List[dict], key: str):
    for i, entry in enumerate(entries):
        print('{}: {}'.format(i, entry[key]))

    choice = int(input('Enter your choice: '))

    if choice < 0 or choice >= len(entries):
        print('Invalid choice')
        exit(1)

    return entries[choice]


if __name__ == '__main__':
    # Experiment chooser
    experiments_args = [
        dict(
            n_episodes=100,
            k=25,
            max_steps=7,
        ),
        dict(
            n_episodes=1000,
            k=2,
            width=12,
            height=4,
            p=[0, .1, .2]
        )
    ]
    experiments = [
        dict(
            title='Deep Sea',
            experiment=DeepSeaExperiment,
            alphas=[.3, .25, .2, .15, 0.1],
            markers=['o', '^', '>', '<', 'v'],
            plot_args=[experiments_args[0]['max_steps'], experiments_args[0]['n_episodes']],
        ),
        dict(
            title='Cliff Walking',
            experiment=CliffWalkingExperiment,
            alphas=[.3, .25, .2, .15, 0.1],
            markers=['o', '^', '>', '<', 'v'],
            plot_args=[experiments_args[1]['width'], experiments_args[1]['height'], experiments_args[1]['n_episodes'],
                       experiments_args[1]['p']],
        )
    ]

    print('Select an old.experiment:')
    choice_experiment = _select_entry(experiments, 'title')
    print('You selected: {}'.format(choice_experiment['title']))
    experiment_constructor = choice_experiment['old.experiment']

    # Agent chooser
    agents = [
        dict(title='Q Learning', agent=QLearning),
        dict(title='Psi Learning', agent=PsiLearning),
        dict(title='G Learning', agent=GLearning),
        dict(title='MIRL', agent=MIRL),
        dict(title='REPS', agent=REPS)
    ]

    plots = [
        dict(choice='Yes', plot=True),
        dict(choice='No', plot=False)
    ]
    print("Do you want to plot the results?")
    choice_plot = _select_entry(plots, 'choice')
    print('You selected: {}'.format(choice_plot['choice']))

    if choice_plot['plot']:
        filenames = [f'{Experiment.DIR_OUTPUT}{experiment_constructor.__name__}_{agent}.npy'
                     for agent in experiment_constructor.AGENTS.keys()]
        alphas = choice_experiment['alphas']
        markers = choice_experiment['markers']
        args = tuple(choice_experiment['plot_args'])
        experiment_constructor.plot(filenames, alphas, markers, *args)

    print('Select an agent:')
    choice_agent = _select_entry(agents, 'title')
    print('You selected: {}'.format(choice_agent['title']))
    agent = choice_agent['agent']

    # Run old.experiment
    print('Running {} old.experiment with {} agent...'.format(choice_experiment['title'], choice_agent['title']))
    index = experiments.index(choice_experiment)
    experiment_args = experiments_args[index]

    # Retrieve agent title label
    agent_title = list(experiment_constructor.AGENTS.keys())[list(experiment_constructor.AGENTS.values()).index(agent)]

    experiment = experiment_constructor([agent_title, agent], **experiment_args)
    experiment.run(write=True)
