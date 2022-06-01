from typing import List

from mushroom_rl.algorithms.value import QLearning

from mdp.algo.model_free.g_learning import GLearning
from mdp.algo.model_free.mirl import MIRL
from mdp.algo.model_free.psi_learning import PsiLearning
from mdp.algo.model_free.reps import REPS
from mdp.experiment.model_free import DeepSeaExperiment, CliffWalkingExperiment, Experiment
from mdp.experiment.model_free_plot import Plotter, AgentsDeepSeaPlotter, AgentsCliffWalkingPlotter, \
    BetasDeepSeaPlotter, BetasCliffWalkingPlotter


class ExperimentRunner:
    AGENTS = [
        dict(title='Q Learning', constructor=QLearning),
        dict(title='Psi Learning', constructor=PsiLearning),
        dict(title='G Learning', constructor=GLearning),
        dict(title='MIRL', constructor=MIRL),
        dict(title='REPS', constructor=REPS)
    ]

    PLOTTERS = [
        dict(
            title='Agents Plotter',
            constructors={
                'Deep Sea (Plot)': AgentsDeepSeaPlotter,
                'Cliff Walking (Plot)': AgentsCliffWalkingPlotter,
            }
        ),
        dict(
            title='Betas Plotter',
            constructors={
                'Deep Sea (Plot)': BetasDeepSeaPlotter,
                'Cliff Walking (Plot)': BetasCliffWalkingPlotter,
            }
        ),
    ]

    @staticmethod
    def prompt_choice(entries: List[dict], key: str) -> int:
        """
        Prompt a choice box where the user can input his selection.

        Args:
            entries: The choice box options
            key: The value of the entries to show

        Returns:
            The index of the selection
        """
        for i, entry in enumerate(entries):
            print('{}: {}'.format(i, entry[key]))

        index = int(input('Enter your choice: '))

        if index < 0 or index >= len(entries):
            print('Invalid choice')
            exit(1)

        print(f'You selected: {entries[index][key]}')
        return index

    @staticmethod
    def choose_experiment(experiments: List[dict]):
        """
        Lets the user select an experiment

        Args:
            experiments: The selectable experiments

        Returns:
            The selected experiment
        """
        print('Select an experiment:')
        index = ExperimentRunner.prompt_choice(experiments, 'title')
        return experiments[index]

    @staticmethod
    def run_experiment(experiment: dict, agent: dict, plot_path: str):
        """
        Runs an experiment with the given agent and store the result.

        Args:
            experiment: The experiment to run
            agent: The agent used in the experiment
            plot_path: The storage path
        """
        # Invoke constructor and run experiment
        experiment_constructor = experiment['constructor']
        agent_constructor = agent['constructor']
        experiment_args = experiment['args']
        experiment_instance = experiment_constructor(agent_constructor, **experiment_args)
        result = experiment_instance.run()

        agent_name = [k for k, v in Experiment.AGENTS_NAME.items() if v == agent['title']][0]
        path = f'{experiment_constructor.__name__}_{agent_name}'
        print('Saving results to {}.npy'.format(path))

        metadata = experiment['args'].copy()
        metadata['agent_full'] = agent['title']
        metadata['agent_short'] = agent_name

        plotter = Plotter(plot_path)
        plotter.save(result, path, **metadata)

    @staticmethod
    def plot(plotter_instance: Plotter, **kwargs):
        """
        Plots the plotters resources.

        Args:
            plotter_instance: The plotter to plots its resources
            kwargs: The additional key-value arguments for the plotter
        """
        filenames = plotter_instance.filenames()
        plotter_instance.plot(filenames, **kwargs)
        plotter_instance.show()

    @staticmethod
    def start_experiment():
        """
        Starts an experiment.
        """
        separator = '-' * 30
        plot_path = 'Alpha_Comparison_01_Cliff'

        experiments = [
            dict(
                title='Deep Sea',
                plot=False,
                constructor=DeepSeaExperiment,
                args=dict(
                    n_episodes=100,
                    k=25,
                    max_steps=4,
                    beta=0.0000000001,
                ),
            ),
            dict(
                title='Cliff Walking',
                plot=False,
                constructor=CliffWalkingExperiment,
                args=dict(
                    n_episodes=1000,
                    k=10,
                    width=12,
                    height=4,
                    p=[0, 0.05, .1, 0.15, .2, 0.25],
                    beta=1,
                ),
            ),
            dict(
                title='Deep Sea (Plot)',
                plot=True,
                args=dict(alphas=[.3, .25, .2, .15, 0.1])
            ),
            dict(
                title='Cliff Walking (Plot)',
                plot=True,
                args=dict(alphas=[.3, .25, .2, .15, 0.1])
            )
        ]

        # Additional arguments
        # Deep Sea
        experiments[0]['args']['steps'] = [i for i in range(1, experiments[0]['args']['max_steps'] + 1)]
        experiments[2]['args'].update(experiments[0]['args'].copy())

        # Cliff Walking
        experiments[1]['args']['steps'] = experiments[1]['args']['p']
        experiments[3]['args'].update(experiments[1]['args'].copy())

        # Experiment choose
        experiment = ExperimentRunner.choose_experiment(experiments)

        print(separator)
        print(f'Running experiment: {experiment["title"]}')
        print(separator)

        if experiment['plot']:
            plotter_constructor = ExperimentRunner.PLOTTERS[
                ExperimentRunner.prompt_choice(ExperimentRunner.PLOTTERS, 'title')
            ]['constructors'][experiment['title']]
            plotter_instance = plotter_constructor(plot_path)
            experiment_args = experiment['args']
            ExperimentRunner.plot(plotter_instance, **experiment_args)
            return

            # Agent chooser
        agent = ExperimentRunner.choose_experiment(ExperimentRunner.AGENTS)

        # Run experiment
        print(separator)
        print(f'Running experiment: {experiment["title"]} with agent {agent["title"]}')
        print(separator)
        ExperimentRunner.run_experiment(experiment, agent, plot_path)


if __name__ == '__main__':
    ExperimentRunner.start_experiment()
