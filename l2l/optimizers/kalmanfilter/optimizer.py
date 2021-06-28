import logging
import numpy as np
import scipy

from collections import namedtuple
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['gamma', 'maxit', 'pop_size', 'seed',
                             'n_iteration', 'stop_criterion', 'sample',
                             'best_n', 'worst_n', 'pick_method',
                             'kwargs'
                             ],
    defaults=(True, 0.25, 0.25, 'random', {'pick_probability': 0.7,
                                           'loc': 0, 'scale': 0.1})
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
:param sampling: bool, If sampling of best individuals should be done
:param best_n: float, Percentage for best `n` individual. In combination with
    `sampling`. Default: 0.25
:param worst_n: float, Percentage for worst `n` individual. In combination with
    `sampling`. Default: 0.25
:param pick_method: str, How the best individuals should be taken. `random` or
    `best_first` must be set. If `pick_probability` is taken then a key
    word argument `pick_probability` with a float value is needed. `gaussian` 
    creates a multivariate normal distribution using the best individuals 
    which will replace the worst individuals. (see also :param kwargs) 
    In combination with `sampling`.
    Default: 'random'.
:param kwargs: dict, key word arguments if `sampling` is True.
    - `pick_probability` - float, probability to pick the first best individual
      Default: 0.7
    - loc - float, mean of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability`
      Default: 0.
    - scale - float, std scale of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability` 
      Default: 0.1
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
        traj.f_add_parameter('sample', parameters.sample,
                             comment='sampling on/off')
        if parameters.sample:
            traj.f_add_parameter('best_n', parameters.best_n,
                                 comment='best n individuals')
            traj.f_add_parameter('worst_n', parameters.worst_n,
                                 comment='worst n individuals')
            traj.f_add_parameter('pick_method', parameters.pick_method,
                                 comment='how to pick random individual')
            traj.f_add_parameter('kwargs', parameters.kwargs,
                                 comment='dict with key word arguments')

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
        if traj.sample:
            ens, model_outs = self.sample_from_individuals(
                individuals=ens,
                model_output=model_outs,
                fitness=fitness,
                sampling_method=traj.sampling_method,
                pick_method=traj.pick_method,
                best_n=traj.best_n,
                worst_n=traj.worst_n,
                **traj.kwargs
            )
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
        if traj.stop_criterion >= self.current_fitness and self.g < traj.n_iteration:
            # Create new individual based on the results of the update from the EnKF.
            new_individual_list = [list_to_dict(results[i],
                                                dict_spec=self.optimizee_individual_dict_spec[i])
                                   for i in range(ensemble_size)]

            # Check this bounding function
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for
                                       ind in new_individual_list]

            fitnesses_results.clear()
            self.eval_pop = new_individual_list
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def sample_from_individuals(self, individuals, fitness, model_output,
                                best_n=0.25, worst_n=0.25,
                                pick_method='random',
                                **kwargs):
        """
        Samples from the best `n` individuals via different methods.
        :param individuals: array_like
            Input data, the individuals
        :param fitness: array_like
            Fitness array
        :param best_n: float
            Percentage of best individuals to sample from
        :param model_output, array like, model outputs of the best indiviudals
            will be used to replace the model outputs of the worst individuals
            in the same manner as the sampling
        :param worst_n:
            Percentage of worst individuals to replaced by sampled individuals
        :param pick_method: str
            Either picks the best individual randomly 'random' or it picks the
            iterates through the best individuals and picks with a certain
            probability `best_first` the first best individual
            `best_first`. In the latter case must be used with the key word
            argument `pick_probability`.  `gaussian` creates a multivariate
            normal using the mean and covariance of the best individuals to
            replace the worst individuals.
            Default: 'random'
        :param kwargs:
            'pick_probability': float
                Probability of picking the first best individual. Must be used
                when `pick_method` is set to `pick_probability`.
            'loc': float, mean of the gaussian normal, can be specified if
               `pick_method` is `random` or `pick_probability`
               Default: 0.
            'scale': float, std scale of the gaussian normal, can be specified if
                `pick_method` is `random` or `pick_probability`
                 Default: 0.1
        :return: array_like
            New array of sampled individuals.
        """
        # best fitness should be here ~ 1 (which means correct choice)
        # sort them from best to worst via the index of fitness
        # get indices
        indices = np.argsort(fitness)[::-1]
        sorted_individuals = np.array(individuals)[indices]
        # get best n individuals from the front
        best_individuals = sorted_individuals[:int(len(individuals) * best_n)]
        # get worst n individuals from the back
        worst_individuals = sorted_individuals[
            len(individuals) - int(len(individuals) * worst_n):]
        # sort model outputs
        sorted_model_output = model_output[indices]
        for wi in range(len(worst_individuals)):
            if pick_method == 'random':
                # pick a random number for the best individuals add noise
                rnd_indx = np.random.randint(len(best_individuals))
                ind = best_individuals[rnd_indx]
                # add gaussian noise
                noise = np.random.normal(loc=kwargs['loc'],
                                         scale=kwargs['scale'],
                                         size=len(ind))
                worst_individuals[wi] = ind + noise
                model_output[wi] = sorted_model_output[rnd_indx]
            elif pick_method == 'best_first':
                for bidx, bi in enumerate(best_individuals):
                    pp = kwargs['pick_probability']
                    rnd_pp = np.random.rand()
                    if pp >= rnd_pp:
                        # add gaussian noise
                        noise = np.random.normal(loc=kwargs['loc'],
                                                 scale=kwargs['scale'],
                                                 size=len(bi))
                        worst_individuals[wi] = bi + noise
                        model_output[wi] = sorted_model_output[bidx]
                        break
            else:
                sampled = self._sample(best_individuals, pick_method)
                worst_individuals = sampled
                rnd_int = np.random.randint(
                    0, len(best_individuals), size=len(best_individuals))
                model_output[len(sorted_individuals) -
                             len(worst_individuals):] = sorted_model_output[rnd_int]
                break
        sorted_individuals[len(sorted_individuals) -
                           len(worst_individuals):] = worst_individuals
        return sorted_individuals, model_output

    def _sample(self, individuals, method='gaussian'):
        if method == 'gaussian':
            dist = Gaussian()
            dist.init_random_state(self.random_state)
            dist.fit(individuals)
            sampled = dist.sample(len(individuals))
        elif method == 'rv_histogram':
            sampled = [scipy.stats.rv_histogram(h) for h in individuals]
        else:
            raise KeyError('Sampling method {} not known'.format(method))
        sampled = np.asarray(sampled)
        return sampled

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
