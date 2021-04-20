import json
import numpy as np
import os
import pandas as pd
import pathlib
import shutil
import subprocess
import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

AntColonyOptimizeeParameters = namedtuple(
    'AntColonyOptimizeeParameters', ['path', 'seed'])


class AntColonyOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.param_path = parameters.path
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.rng = np.random.default_rng(parameters.seed)
        self.dir_path = ''
        fp = pathlib.Path(__file__).parent.absolute()
        print(os.path.join(str(fp), 'config.json'))
        with open(
                os.path.join(str(fp), 'config.json')) as jsonfile:
            self.config = json.load(jsonfile)

    def create_individual(self):
        """
        Creates and returns the individual

        Creates the parameters for netlogo.
        The parameter are `weights`, and `delays`.
        """
        # TODO the creation of the parameters should be more variable
        #  e.g. as parameters or as a config file
        # create random weights
        weights = self.rng.uniform(-20, 20, 220)
        # create delays
        delays = self.rng.integers(low=1, high=7, size=220)
        # create individual
        individual = {
            'weights': weights,
            'delays': delays
        }
        return individual

    def simulate(self, traj):
        """
        Simulate a run and return a fitness

        A directory `individualN` and a csv file `individualN` with parameters
        to optimize will be saved.
        Invokes a run of netlogo, reads in a file outputted (`resultN`) by
        netlogo with the fitness inside.
        """
        weights = traj.individual.weights
        delays = traj.individual.delays
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        # create directory individualN
        self.dir_path = os.path.join(self.param_path,
                                     'individual{}'.format(self.ind_idx))
        try:
            os.mkdir(self.dir_path)
        except FileExistsError:
            shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)

        individual = {
            'weights': weights,
            'delays': delays.astype(int)
        }
        # create the csv file and save it in the created directory
        df = pd.DataFrame(individual)
        df = df.T
        df.to_csv(os.path.join(self.dir_path, 'individual_config.csv'.format(self.ind_idx)),
                  header=False, index=False)
        # get paths etc. from config file
        model_path = self.config['model_path']
        model_name = self.config['model_name']
        headless_path = self.config['netlogo_headless_path']
        # copy model to the created directory
        shutil.copyfile(model_path, os.path.join(self.dir_path, model_name))
        # call netlogo
        subdir_path = os.path.join(self.dir_path, model_name)
        try:
            subprocess.run(['bash', '{}'.format(headless_path),
                            '--model', '{}'.format(subdir_path),
                            '--experiment', 'experiment1',
                            '--table', 'table1.csv'])
        except subprocess.CalledProcessError as cpe:
            print('Optimizee process error {}'.format(cpe))
        file_path = os.path.join(self.dir_path, "individual_result.csv")
        while not os.path.isfile(file_path):
            time.sleep(5)
        # Read the results file after the netlogo run
        csv = pd.read_csv(file_path, header=None, na_filter=False)
        # We need the last row and first column
        fitness = csv.iloc[-1][0]
        print('Fitness {}in generation {} individual {}'.format(
            fitness,
            self.generation,
            self.ind_idx))
        # remove directory
        shutil.rmtree(self.dir_path)
        return (fitness,)

    def bounding_func(self, individual):
        return individual