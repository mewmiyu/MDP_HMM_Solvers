from typing import List

from mushroom_rl.algorithms.value import QLearning

from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.reps import REPS
from mdp.experiment.model_free import DeepSeaExperiment, CliffWalkingExperiment, Experiment
from mdp.experiment.model_free_plot import AgentsDeepSeaPlotter, AgentsCliffWalkingPlotter, \
    BetasCliffWalkingPlotter, BetasDeepSeaPlotter, Plotter


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
            max_steps=3,
            beta=0.00000001,
            plot_args=dict(
                alphas=[.3, .25, .2, .15],
                markers=['o', '^', '>', '<'],
                betas=[0.00000001, 0.000001, 0.0001, 0.00000001]
            ),
        ),
        dict(
            title='Cliff Walking',
            constructor=CliffWalkingExperiment,
            n_episodes=1000,
            k=10,
            width=12,
            height=4,
            p=[0, .1, .2],
            beta=0.1,
            plot_args=dict(
                alphas=[.3, .25, .2, .15],
                markers=['o', '^', '>', '<'],
                betas=[]
            ),
        )
    ]
    experiments[0]['steps'] = [i for i in range(1, experiments[0]['max_steps'] + 1)]
    experiments[1]['steps'] = experiments[1]['p']

    plotters = [
        dict(
            title='Agents Plotter',
            constructors=dict(
                deep_sea=AgentsDeepSeaPlotter,
                cliff_walking=AgentsCliffWalkingPlotter
            )
        ),
        dict(
            title='Betas Plotter',
            constructors=dict(
                deep_sea=BetasDeepSeaPlotter,
                cliff_walking=BetasCliffWalkingPlotter
            )
        ),
    ]

    # Experiment chooser
    print('Select an experiment:')
    size = len(experiments)
    for i, experiment_kwargs in enumerate(experiments):
        print('{}: {}'.format(i, experiment_kwargs['title']))
    for i, experiment_kwargs in enumerate(experiments):
        print('{}: {} (plot)'.format(i + size, experiment_kwargs['title']))

    choice_experiment = int(input('Enter your choice: '))
    plot = choice_experiment > size - 1
    if choice_experiment > size - 1:
        choice_experiment -= size
    experiment_kwargs = experiments[choice_experiment]
    experiment_constructor = experiment_kwargs['constructor']

    print(separator)
    print('Running experiment: {} (plot={})'.format(experiment_kwargs['title'], plot))
    print(separator)

    # Plotter chooser
    if plot:
        print('Select a plotter:')
        plotter = _select_entry(plotters, 'title')
        plotter_constructor = list(plotter['constructors'].values())[choice_experiment]
        _plotter = plotter_constructor(plot_path)
        print(type(_plotter))
        filenames = _plotter.filenames()
        print(filenames)
        _plotter.plot(filenames, **experiment_kwargs)
        _plotter.show()
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
        print('Running {} experiment with {} agent...'.format(experiment_kwargs['title'], choice_agent['title']))
        # Retrieve agent title label
        agent_title = list(Experiment.AGENTS.keys())[list(Experiment.AGENTS.values()).index(agent)]
        print('Agent title: {}'.format(agent_title))
        experiment = experiment_constructor(agent, **experiment_kwargs)
        result = experiment.run()

        if agent == QLearning:
            path = f'{experiment_constructor.__name__}_{agent_title}'
        else:
            path = f'{experiment_constructor.__name__}_beta{experiment_kwargs["beta"]}_{agent_title}'

        print('Saving results to {}.npy'.format(path))

        _plotter = Plotter(plot_path)
        _plotter.save(path, result)