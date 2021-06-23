from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.tvb.optimizee import TVBOptimizee
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters


def main():
    name = 'L2L-TVB-PYTHON-GRID'
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = TVBOptimizee(traj, 0)

    ## Outerloop optimizer initialization
    parameters = GridSearchParameters({'coupling': [0.0, 3.0, 1], 'speed': [1.0, 2.0, 1]})

    optimizer = GridSearchOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1.0,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    # Experiment run
    experiment.run_experiment(optimizee=optimizee, optimizer=optimizer,
                              optimizee_parameters=parameters)
    # End experiment
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
