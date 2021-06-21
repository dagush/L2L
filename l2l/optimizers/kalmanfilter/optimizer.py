import logging

import numpy as np

from collections import namedtuple
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['gamma', 'maxit', 'pop_size', 'seed',
                             'n_iteration', 'stop_criterion'],
)

EnsembleKalmanFilterParameters.__doc__ = """
:param gamma: float, A small value, multiplied with the eye matrix
:param maxit: int, Epochs to run inside the Kalman Filter
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
    Corresponds to number of ensembles
:param seed: The random seed used to sample and fit the distribution. 
    Uses a random generator seeded with this seed.
:param stop_criterion: float, When the max. current fitness is bigger or equal 
    the `stop_criterion` the optimization ends
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.parameters = parameters

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        traj.f_add_parameter('maxit', parameters.maxit,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Seed used for random number generation '
                                     'in optimizer')
        traj.f_add_parameter('pop_size', parameters.pop_size)
        traj.results.f_add_result_group('generation_params')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion,
                             comment='stopping threshold')

        # Set the random state seed for distribution
        self.random_state = np.random.RandomState(traj.parameters.seed)

        self.optimizee_individual_dict_spec = []
        for i in range(parameters.pop_size):
            _, dict_spec = dict_to_list(
                self.optimizee_create_individual(), get_dict_spec=True)
            self.optimizee_individual_dict_spec.append(dict_spec)

        current_eval_pop = [dict_to_list(self.optimizee_create_individual(),
                                         get_dict_spec=False) for _ in
                            range(parameters.pop_size)]

        self.eval_pop = [list_to_dict(current_eval_pop[i], self.optimizee_individual_dict_spec[i])
                         for i in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            self.eval_pop = [self.optimizee_bounding_func(ind) for ind in
                             self.eval_pop]

        self.current_fitness = -np.inf
        self.best_individual = {'generation': 0,
                                'individual': 0,
                                'fitness': self.current_fitness}
        # TODO create observations
        self.observations = [50]
        self.g = 0

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        self.eval_pop.clear()

        individuals = traj.individuals[self.g]
        ensemble_size = traj.pop_size
        weights = [dict_to_list(individuals[i].params, get_dict_spec=False)
                   for i in range(ensemble_size)]
        fitness = np.squeeze(list(dict(fitnesses_results).values()))
        self.current_fitness = np.max(fitness)
        if self.current_fitness > self.best_individual['fitness']:
            self.best_individual['fitness'] = self.current_fitness
            self.best_individual['individual'] = np.argmax(fitness)
            self.best_individual['generation'] = self.g
        ens = np.array(weights)
        # sort from best to worst
        best_indviduals = np.argsort(fitness)[::-1]
        current_res = np.sort(fitness)[::-1]
        logger.info('Sorted Fitness best to worst {}'.format(current_res))
        logger.info(
            'Best fitness {} in generation {}'.format(self.current_fitness,
                                                      self.g))
        logger.info('Best individuals index {}'.format(best_indviduals))
        logger.info('Mean of individuals {}'.format(np.mean(current_res)))
        model_outs = fitness.reshape((ensemble_size, len(self.observations), -1))
        self.observations = np.array(self.observations * model_outs.shape[-1])
        gamma = np.eye(self.observations.ndim) * traj.gamma
        enkf = EnKF(maxit=traj.maxit)
        enkf.fit(ensemble=ens,
                 ensemble_size=ensemble_size,
                 observations=self.observations,
                 model_output=model_outs,
                 gamma=gamma)
        # These are all the updated weights for each ensemble
        results = enkf.ensemble.cpu().numpy()

        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)

        generation_result_dict = {
            'generation': traj.generation,
            'weights': results
        }
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        # Produce the new generation of individuals
        if traj.stop_criterion >= self.current_fitness or self.g < traj.n_iteration:
            # Create new individual based on the results of the update from the EnKF.
            new_individual_list = [list_to_dict(results[i],
                                                dict_spec=self.optimizee_individual_dict_spec[i])
                                   for i in range(ensemble_size)]
        else:
            new_individual_list = []

            # Check this bounding function
        if self.optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for
                                   ind in new_individual_list]

        fitnesses_results.clear()
        self.eval_pop = new_individual_list
        self.g += 1  # Update generation counter
        self._expand_trajectory(traj)

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.
        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`
        """
        traj.f_add_result('final_individual', self.best_individual)
        logger.info(
            "The best individuals with fitness {}".format(
                self.best_individual))
        logger.info("-- End of (successful) EnKF optimization --")
