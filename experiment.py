from typing import List

from mushroom_rl.algorithms.value import QLearning

from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.reps import REPS
from mdp.experiment.model_free import DeepSeaExperiment, CliffWalkingExperiment, Experiment
from mdp.experiment.model_free_plot import AgentsPlotter, AgentsDeepSeaPlotter, AgentsCliffWalkingPlotter


def _select_entry(entries: List[dict], key: str):
    for i, entry in enumerate(entries):
        print('{}: {}'.format(i, entry[key]))

    choice = int(input('Enter your choice: '))

    if choice < 0 or choice >= len(entries):
        print('Invalid choice')
        exit(1)

    return entries[choice]


if __name__ == '__main__':
    separator = '-' * 30
    plot_path = ''
    experiments = [
        dict(
            title='Deep Sea',
            constructor=DeepSeaExperiment,
            n_episodes=100,
            k=25,
            max_steps=7,
            plotter_constructor=AgentsDeepSeaPlotter,
            plot=False,
            plot_args=dict(
                alphas=[.3, .25, .2, .15, 0.1],
                markers=['o', '^', '>', '<', 'v'],
            )
        ),
        dict(
            title='Cliff Walking',
            constructor=CliffWalkingExperiment,
            n_episodes=1000,
            k=2,
            width=12,
            height=4,
            p=[0, .1, .2],
            plotter_constructor=AgentsCliffWalkingPlotter,
            plot=False,
            plot_args=dict(
                alphas=[.3, .25, .2, .15, 0.1],
                markers=['o', '^', '>', '<', 'v'],
            ),
        )
    ]
    experiments[0]['steps'] = [i for i in range(1, experiments[0]['max_steps'] + 1)]
    experiments[1]['steps'] = experiments[1]['p']

    # Experiment chooser
    print('Select an experiment:')
    size = len(experiments)
    for i, experiment in enumerate(experiments):
        print('{}: {}'.format(i, experiment['title']))
    for i, experiment in enumerate(experiments):
        print('{}: {} (plot)'.format(i + size, experiment['title']))

    choice_experiment = int(input('Enter your choice: '))
    plot = choice_experiment > size - 1
    if choice_experiment > size - 1:
        choice_experiment -= size
    experiment = experiments[choice_experiment]
    experiment_constructor = experiment['constructor']
    experiment['plot'] = plot

    print(separator)
    print('Running experiment: {} (plot={})'.format(experiment['title'], experiment['plot']))
    print(separator)

    # Plotter chooser
    plotter = experiment['plotter_constructor'](plot_path)
    if experiment['plot']:
        filenames = [f'{plotter.path}{experiment_constructor.__name__}_{agent}.npy'
                     for agent in Experiment.AGENTS.keys()]
        experiment['titles'] = [agent for agent in Experiment.AGENTS.keys()]

        plotter.plot(filenames, **experiment)
        plotter.show()
    else:
        # Agent chooser
        agents = [
            dict(title='Q Learning', agent=QLearning),
            dict(title='Psi Learning', agent=PsiLearning),
            dict(title='G Learning', agent=GLearning),
            dict(title='MIRL', agent=MIRL),
            dict(title='REPS', agent=REPS)
        ]
        print('Select an agent:')
        choice_agent = _select_entry(agents, 'title')
        agent = choice_agent['agent']

        print(separator)
        print('You selected: {}'.format(choice_agent['title']))
        print(separator)

        # Run experiment
        print('Running {} experiment with {} agent...'.format(experiment['title'], choice_agent['title']))
        # Retrieve agent title label
        agent_title = list(Experiment.AGENTS.keys())[list(Experiment.AGENTS.values()).index(agent)]
        print('Agent title: {}'.format(agent_title))
        experiment = experiment_constructor(agent, **experiment)
        result = experiment.run()

        path = f'{experiment_constructor.__name__}_{agent_title}'
        print('Saving results to {}.npy'.format(path))
        plotter.save(path, result)
