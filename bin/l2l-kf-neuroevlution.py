from datetime import datetime
from l2l.utils.experiment import Experiment

from l2l.optimizees.neuroevolution import NeuroEvolutionOptimizee, NeuroEvolutionOptimizeeParameters
from l2l.optimizers.kalmanfilter import EnsembleKalmanFilterParameters, EnsembleKalmanFilter


def run_experiment():
    experiment = Experiment(
        root_dir_path='../results')
    jube_params = {"exec": "python"}
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name="NeuroEvo_KF_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))

    # Optimizee params
    optimizee_parameters = NeuroEvolutionOptimizeeParameters(
        path=experiment.root_dir_path, seed=1)
    optimizee = NeuroEvolutionOptimizee(traj, optimizee_parameters)

    optimizer_seed = 1234
    optimizer_parameters = EnsembleKalmanFilterParameters(gamma=0.01,
                                                          maxit=1,
                                                          pop_size=2,
                                                          n_iteration=3,
                                                          seed=optimizer_seed,
                                                          stop_criterion=50
                                                          )

    optimizer = EnsembleKalmanFilter(traj,
                                     optimizee_create_individual=optimizee.create_individual,
                                     optimizee_fitness_weights=(-0.1,),
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
