from datetime import datetime
from l2l.utils.experiment import Experiment

from l2l.optimizees.neuroevolution import AntColonyOptimizee, AntColonyOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer


def run_experiment():
    experiment = Experiment(
        root_dir_path='../results')
    jube_params = {"exec": "python"}
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name="AC_GA_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))

    # Optimizee params
    optimizee_parameters = AntColonyOptimizeeParameters(
        path=experiment.root_dir_path, seed=1)
    optimizee = AntColonyOptimizee(traj, optimizee_parameters)

    optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=4,
                                                      cx_prob=0.7,
                                                      mut_prob=0.5,
                                                      n_iteration=3,
                                                      ind_prob=0.02,
                                                      tourn_size=15,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters)
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
