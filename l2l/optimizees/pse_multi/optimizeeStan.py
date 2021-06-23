import logging

import numpy
from sdict import sdict
import random

from l2l.optimizees.optimizee import Optimizee

from read_csvs import read_samples

logger = logging.getLogger("ltl-pse")


class PSEOptimizee(Optimizee):

    def __init__(self, trajectory, seed=27):

        super(PSEOptimizee, self).__init__(trajectory)
        seed = numpy.uint32(seed)
        self.random_state = numpy.random.RandomState(seed=seed)

    def simulate(self, trajectory):

        self.id = trajectory.individual.ind_idx
        print(self.id)
        self.tau0 = trajectory.individual.tau0
        f = open("/p/project/cslns/vandervlag1/cmdstan/examples/bayes/paradump_{}.txt".format(self.id), "w")
        for c in self.tau0:
            f.write("{} ".format(c))
        f.write("\n")

        proc = subprocess.Popen(['/p/project/cslns/vandervlag1/cmdstan/examples/bernoulli/bernoulli', 'sample',
                                 'data', 'file=/p/project/cslns/vandervlag1/cmdstan/examples/bernoulli/bernoulli.data.json'])

        proc.wait()

        self.fitness = []
        dict_samples_loglik = read_samples(filecsv, nwarmup=0, nsampling=num_samples, variables_of_interest=['log_lik'])
        self.fitness = numpy.sum(dict_samples_loglik)

        return self.fitness

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """

        bound_tau0 = [10.0, 100.0]
        tau0_array = []
        num_of_parameters = 16
        for i in range(num_of_parameters):
            tau0_array.append(random.uniform(bound_tau0[0], bound_tau0[1]))
        tau0_dict = {'tau0': tau0_array }
        return tau0_dict


    def bounding_func(self, individual):
        return individual


def end(self):
    logger.info("End of all experiments. Cleaning up...")
    # There's nothing to clean up though


def main():
    import yaml
    import os
    import logging.config

    from ltl import DummyTrajectory
    from ltl.paths import Paths
    from ltl import timed

    # TODO: Set root_dir_path here
    paths = Paths('pse', dict(run_num='test'), root_dir_path='.')  # root_dir_path='.'

    fake_traj = DummyTrajectory()
    optimizee = PSEOptimizee(fake_traj)
    params = optimizee.create_individual()
    params['generation']=0
    params['ind_idx'] = 0
    fake_traj.individual = sdict(params)

    testing_error = optimizee.simulate(fake_traj)
    print("Testing error is ", testing_error)


if __name__ == "__main__":
    main()
