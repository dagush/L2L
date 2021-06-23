import logging

from collections import namedtuple

import numpy
from sdict import sdict

from l2l.optimizees.optimizee import Optimizee
import subprocess
# from utils.individual import Individual

import random
from l2l.optimizees.optimizee.read_csvs import read_samples
# import l2l.optimizees.optimizee.stan.read_csvs


logger = logging.getLogger("ltl-sp")

# BayesOptimizeeParameters = namedtuple('BayesOptimizeeParameters', ['tau0'])

class BayesOptimizee(Optimizee):

    def __init__(self, trajectory, seed=27):

        super(BayesOptimizee, self).__init__(trajectory)
        # If needed
        seed = numpy.uint32(seed)
        self.random_state = numpy.random.RandomState(seed=seed)

    def simulate(self, trajectory):
        self.id = trajectory.individual.ind_idx
        print(self.id)
        # self.delay = trajectory.individual.delay
        # self.coupling = trajectory.individual.coupling
        self.tau0 = trajectory.individual.tau0
        # self.init_(trajectory)

        # Dump the parameters to the input file
        # f = open("input_{}.txt".format(self.id), "w")
        # f = open("/p/project/cslns/vandervlag1/L2L/bin/results/DatainputFullSeeg_ODE_config_l2l_{}.R".format(self.id), "w")
        # Maybe it is a list of couplings, so we need to go through all of them
        #for c in self.delay:
        #    for d in self.coupling:
        #        f.write("{},{}\n".format(c, d)) # we have to delete the old values

        # for c in self.tau0:
        #     f.write("{} ".format(c))
        # f.write("\n")
        # for d in self.coupling:
        #     f.write("{} ".format(d))
        # f.close()

        # Assuming that the program sends to stdout only the output of its fitness:
        # proc = subprocess.Popen("<program_name> input_{}.txt result_{}.txt".format(self.id,self.id), shell=True)
        # input file also .txt ?
        # proc = subprocess.Popen("/p/project/cjinm71/Wischnewski/C++/KuramotoCPU.cpp.lx \

        data_input = "/p/project/cslns/vandervlag1/cmdstan/examples/bayes/DatainputFullSeeg_ODE_config_l2l_{}.R".format(self.id)
        with open(data_input, "a+") as file_object:
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            # Append text at the end of file
            file_object.write('tau0 <- ' + self.tau0)

        proc = subprocess.Popen(['/p/project/cslns/vandervlag1/cmdstan/examples/bernoulli/bernoulli', 'sample',
                                 'data', 'file=/p/project/cslns/vandervlag1/cmdstan/examples/bernoulli/bernoulli.data.json'])
                                # stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # proc = subprocess.Popen("/p/project/cslns/vandervlag1/cmdstan/bernouille.sh \
        #     /p/project/cslns/vandervlag1/L2L/bin/results/paradump_{}.txt")
            # /p/project/cslns/vandervlag1/L2L/bin/results/result_{}.txt".format(self.id, self.id), shell=True)
            # /p/project/cjinm71/Wischnewski/TVB/101309Subj_GPU03_KW_F/input_Kuramoto_CPU_A9 \
            # /p/project/cjinm71/Wischnewski/TVB/101309Subj_GPU03_KW_F/Werte_{}.txt \
            # /p/project/cjinm71/Wischnewski/TVB/101309Subj_GPU03_KW_F/Result_{}.txt".format(self.id, self.id), shell=True)

        proc.wait()
        # Result was dumped to file Result.txt
        # self.fitness = []
        # with open("/p/project/cslns/vandervlag1/L2L/bin/results/result_{}.txt".format(self.id), "r") as f:
        # # with open("/p/project/cslns/vandervlag1/L2L/bin/results/result.txt", "r") as f:
        #     line = f.readline()
        #     while line:
        #         self.fitness.extend([float(line)])
        #         # line = f.readline()

        self.fitness = []
        filecsv = "/p/project/cslns/vandervlag1/cmdstan/examples/bayes/data_output_hmc_Seeghorseshoe2/output_hmc_Seeghorseshoe2_{}.csv".format(self.id)
        num_samples = 2
        dict_samples_loglik = read_samples(filecsv, nwarmup=0, nsampling=num_samples, variables_of_interest=['log_lik'])
        self.fitness = numpy.sum(dict_samples_loglik)

        #proc = subprocess.Popen("rm /p/project/cjinm71/Wischnewski/TVB/101309Subj_GPU03_KW_F/Werte_{}.txt".format(self.id))
        return numpy.array(self.fitness)


    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        # self.bound_gr = [0,0] # for delay
        # self.bound_gr[0] = 0
        # self.bound_gr[1] = 94
        #
        # self.bound_gr2 = [0,0] # for coupling
        # self.bound_gr2[0] = 0
        # self.bound_gr2[1] = 0.945
        #
        # num_of_parameters=16 # 48
        # #return{'coupling':[5,6,7,8,9], 'delay':[0.1,1,10,12]}
        # delay_array = []
        # coupling_array = []
        # for i in range(num_of_parameters):
        #     delay_array.extend([self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0]])
        #     coupling_array.extend([self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]])
        #
        # print(delay_array)
        # print(coupling_array)
        # return {'delay': delay_array, 'coupling':coupling_array}
        #return {'delay': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0],
        #      'coupling': self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]}

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
    paths = Paths('pse', dict(run_num='test'), root_dir_path='/p/project/cslns/vandervlag1') # root_dir_path='.'

    fake_traj = DummyTrajectory()
    optimizee = BayesOptimizee(fake_traj)
    #ind = Individual(generation=0,ind_idx=0,params={})
    params  =optimizee.create_individual()
    #params['generation']=0
    params['ind_idx']=0
    #fake_traj.f_expand(params)
    #for key,val in params.items():
    #    ind.f_add_parameter(key, val)
    fake_traj.individual = sdict(params)
    #fake_traj.individual.ind_idx = 0

    testing_error = optimizee.simulate(fake_traj)
    print("Testing error is ", testing_error)

if __name__ == "__main__":
    main()
