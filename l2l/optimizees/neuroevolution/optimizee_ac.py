import json
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import shutil
import subprocess
import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

AntColonyOptimizeeParameters = namedtuple(
    'AntColonyOptimizeeParameters', ['path', 'seed', 'save_n_generation',
                                     'run_headless', 'load_parameter'])


class AntColonyOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.param_path = parameters.path
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.save_n_generation = parameters.save_n_generation
        self.rng = np.random.default_rng(parameters.seed)
        self.dir_path = ''
        self.fp = pathlib.Path(__file__).parent.absolute()
        self.is_headless = parameters.run_headless
        self.load_parameter = parameters.load_parameter
        print(os.path.join(str(self.fp), 'config.json'))
        with open(
                os.path.join(str(self.fp), 'config.json')) as jsonfile:
            self.config = json.load(jsonfile)

    def create_individual(self):
        """
        Creates and returns the individual

        Creates the parameters for netlogo.
        The parameter are `weights`, and `delays`.
        """
        if self.load_parameter:
            weights, delays = self.reload_parameter()
        else:
            # TODO the creation of the parameters should be more variable
            #  e.g. as parameters or as a config file
            # create random weights
            weights = self.rng.uniform(-20, 20, 250)
            # create delays
            delays = self.rng.integers(low=1, high=7, size=250)
        # create individual
        individual = {
            'weights': weights,
            'delays': np.round(delays).astype(int)
        }
        return individual

    def reload_parameter(self):
        high = 9  # corresponds to pop_size, could be given as parameter
        randint = self.rng.integers(0, high, 1)[0]
        if self.load_parameter:
            traj_path = os.path.join(
                self.param_path, 'traj_test/simulation/trajectories')
            # create a function to split strings via "_" and take 2. argument
            # it also remove the .bin suffix
            def func(x, idx=2): return int(x[:-4].split('_')[idx])
            traj_ids = [func(f) for f in os.listdir(
                traj_path) if f.endswith('.bin')]
            traj_ids = np.sort(np.unique(traj_ids))
            # get the previous last generation idx as sometimes in the latest
            # generation there is a crash and not all individuals are created
            # if only one generation was iterated take the latest one
            # if len(traj_ids) > 1:
            #     last_gen_idx = traj_ids[-2]
            # else:
            #     last_gen_idx = traj_ids[-1]
            last_gen_idx = traj_ids[-1]
            # last_gen_idx = func(sorted(trajs, key=func)[::-1][0])
            with open(f'{traj_path}/trajectory_{randint}_{last_gen_idx}.bin',
                      'rb') as tr:
                print(
                    f'loading trajectories {traj_path}/trajectory_{randint}_{last_gen_idx}.bin')
                trajectory = pickle.load(tr)
            weights = trajectory.individual.weights
            delays = trajectory.individual.delays
        return weights, delays

    def simulate(self, traj):
        """
        Simulate a run and return a fitness

        A directory `individualN` and a csv file `individualN` with parameters
        to optimize will be saved.
        Invokes a run of netlogo, reads in a file outputted (`resultN`) by
        netlogo with the fitness inside.
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        weights = traj.individual.weights
        delays = traj.individual.delays

        # create directory individualN
        self.dir_path = os.path.join(self.param_path,
                                     f'individual_{self.ind_idx}')
        try:
            os.mkdir(self.dir_path)
        except FileExistsError as fe:
            print(fe)
            shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)

        individual = {
            'weights': weights,
            'delays': np.round(delays).astype(int)
        }
        # create the csv file and save it in the created directory
        df = pd.DataFrame(individual)
        df = df.T
        df.to_csv(os.path.join(self.dir_path, f'individual_config.csv'),
                  header=False, index=False)
        # get paths etc. from config file
        model_path = self.config['model_path']
        model_name = self.config['model_name']
        headless_path = self.config['netlogo_headless_path']
        # Full model path with name
        model = os.path.join(model_path, model_name)
        # copy model to the created directory
        shutil.copyfile(model, os.path.join(self.dir_path, model_name))
        # call netlogo
        subdir_path = os.path.join(self.dir_path, model_name)
        python_file = os.path.join(self.fp, self.config['pynetlogo_model'])
        try:
            if self.is_headless:
                subp = subprocess.Popen(['bash', '{}'.format(headless_path),
                                         '--model', '{}'.format(subdir_path),
                                        '--experiment', 'experiment1',
                                         '--threads', '1'],
                                        shell=False)
            else:
                subp = subprocess.Popen(['python', f'{python_file}',
                                         '--netlogo_home', f'{self.config["netlogo_home"]}',
                                        '--netlogo_version', f'{self.config["netlogo_version"]}',
                                         '--model', f'{subdir_path}',
                                         '--ticks', '10000',
                                         '--individual_no', f'{self.ind_idx}'
                                         ],
                                        shell=False)
        except subprocess.CalledProcessError as cpe:
            print('Optimizee process error {}'.format(cpe))
        file_path = os.path.join(
            self.dir_path, "individual_result.csv")
        while True:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    line = file.read()
                    if line:
                        break
            time.sleep(5)
        subp.kill()
        # Read the results file after the netlogo run
        csv = pd.read_csv(file_path, header=None, na_filter=False)
        # We need the last row and first column
        fitness = csv.iloc[-1][0]
        print('Fitness {} in generation {} individual {}'.format(fitness,
                                                                 self.generation,
                                                                 self.ind_idx))
        # save every n generation the results
        if self.generation % self.save_n_generation == 0:
            # create folder if not existent
            result_folder = os.path.join(self.param_path, 'results')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            # rename to individual_GEN_INDEX_results.csv
            results_filename = "individual_{}_{}_result.csv".format(
                self.generation, self.ind_idx)
            shutil.copyfile(file_path, os.path.join(
                result_folder, results_filename))
        # remove directory
        shutil.rmtree(self.dir_path)
        return (fitness,)

    def bounding_func(self, individual):
        # clip the params
        individual = {"weights": np.clip(individual['weights'], -20, 20),
                      "delays": np.clip(individual['delays'], 1, 5)}
        return individual
