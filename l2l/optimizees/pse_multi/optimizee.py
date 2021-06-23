import logging

import numpy as np
from sdict import sdict

from l2l.optimizees.optimizee import Optimizee
import subprocess
import pickle

logger = logging.getLogger("ltl-pse")

# class SubmitJob(gc3libs.Application)

class PSEOptimizee(Optimizee):

    def __init__(self, trajectory, seed=27):

        super(PSEOptimizee, self).__init__(trajectory)
        # If needed
        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

    def simulate(self, trajectory):
        self.id = trajectory.individual.ind_idx
        self.delay = trajectory.individual.delay
        self.coupling = trajectory.individual.coupling

        # Pickle the L2L produced parameters such that your application can pick them up
        coup_file = open('/p/project/cslns/wikicollab/Parsweep_L2L_RateML/couplings_%d' % self.id, 'wb')
        dela_file = open('/p/project/cslns/wikicollab/Parsweep_L2L_RateML/speeds_%d' % self.id, 'wb')
        pickle.dump(self.coupling, coup_file)
        pickle.dump(self.delay, dela_file)
        coup_file.close()
        dela_file.close()

        # Start the to optimize process which can be any executable
        # Make sure to read in the pickled data from L2L
        # Set the rateML execution and results folder on your system
        # TODO: make nicer
        try:
            subprocess.run(['python', 'rateML/parsweep.py',
                            '--model', 'oscillator',
                            '-c32', '-s32', '-n400',
                            '--tvbn', '76', '--stts', '2',
                            '--procid', str(self.id)], check=True)
        except subprocess.CalledProcessError:
            logger.error('Optimizee process error')

        # Results are dumped to file result_[self.id].txt. Unpickle them here
        self.fitness = []
        cuda_RateML_res_file = open('rateML/result_%d' % self.id, 'rb')
        self.fitness = pickle.load(cuda_RateML_res_file)
        cuda_RateML_res_file.close()

        # self.fitness = []
        # if id == 0:
        #     filename = "/p/project/cslns/vandervlag1/L2L/results/Result_0.txt"
        # else:
        #     filename = "/p/project/cslns/vandervlag1/L2L/results/Result_1.txt"
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     while line:
        #         self.fitness.extend([line])
        #         line = f.readline()
        # [0.1, 2.6, ....fit_1023]
        return self.fitness

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        # self.bound_gr = [0, 0]  # for delay
        # self.bound_gr[0] = 0
        # self.bound_gr[1] = 94
        #
        # self.bound_gr2 = [0, 0]  # for coupling
        # self.bound_gr2[0] = 0
        # self.bound_gr2[1] = 0.945

        #num_of_parameters = 32  # will produce 32*32 parameter combinations
        # return{'coupling':[5,6,7,8,9], 'delay':[0.1,1,10,12]}
        # delay_array = []
        # coupling_array = []
        # for i in range(num_of_parameters):
        #     delay_array.extend([self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0]])
        #     coupling_array.extend(
        #         [self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]])

        # coupling_array = np.linspace(0.003, 0.005, num_of_parameters)
        # delay_array = np.linspace(3.0, 5.0, num_of_parameters)

        # print(delay_array)
        # print(coupling_array)
        # return {'delay': delay_array, 'coupling': coupling_array}
        # return {'delay': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0],
        #      'coupling': self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]}
        return {'delay': (np.random.rand()*(5.0-3.0))+3.0, 'coupling': (np.random.rand()*(0.005-0.004))+0.004}

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
    # ind = Individual(generation=0,ind_idx=0,params={})
    params = optimizee.create_individual()
    # params['generation']=0
    params['ind_idx'] = 0
    # fake_traj.f_expand(params)
    # for key,val in params.items():
    #    ind.f_add_parameter(key, val)
    fake_traj.individual = sdict(params)
    # fake_traj.individual.ind_idx = 0

    testing_error = optimizee.simulate(fake_traj)
    print("Testing error is ", testing_error)


if __name__ == "__main__":
    main()
