import logging
from sdict import sdict
from l2l.optimizees.optimizee import Optimizee
import tvb.simulator.lab as lab
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("ltl-tvb")


class TVBOptimizee(Optimizee):

    def __init__(self, trajectory, seed):
        super(TVBOptimizee, self).__init__(trajectory)

    # Calculates the Functional Connectivity
    def compFC(self, ys):
        return np.corrcoef(ys.T)

    # Calculates the degree of correlation between the Functional and the Structural Connectivity
    def corrSCFC(self, SC, FCloc):
        return np.corrcoef(FCloc.ravel(), SC.ravel())[0, 1]

    # Generates a plot of the Functional and Structural connectivity. Returns their correlation coefficient.
    def plot_FCSC(self, SC, FCloc, i):
        fig, ax = plt.subplots(ncols=2, figsize=(12, 3))

        sns.heatmap(FCloc, xticklabels='',
                    yticklabels='', ax=ax[0],
                    cmap='coolwarm', vmin=-1, vmax=1)

        sns.heatmap(SC / SC.max(), xticklabels='', yticklabels='',
                    ax=ax[1], cmap='coolwarm', vmin=-1, vmax=1)

        ax[0].set_title('simulated FC. \nSC-FC r = ' + str(i))
        ax[1].set_title('SC')
        #plt.show()
        plt.savefig("FC vs SC")

    # Generates a new TVB model. This includes connectivity, integration scheme and neural mass model.

    def tvb_model(self):
        # Neural mass model
        populations = lab.models.ReducedWongWang()  # RWW
        # Connectivity definition based on DTI data
        white_matter = lab.connectivity.Connectivity.from_file()
        print(self.coupling)
        white_matter.configure()
        # Deffinition of the coupling filter between regions
        white_matter_coupling = lab.coupling.Linear(a=np.array(self.coupling))
        # Defines noise if it is used
        # noise_ = lab.noise.Additive(nsig=np.array(2 ** -6))
        # Specify the integrator.
        heunint = lab.integrators.EulerDeterministic(
            dt=0.1)  # EulerStochastic(dt=1.0,noise=noise_)
        return populations, white_matter, white_matter_coupling, heunint

    # Performs one simulation and returns the results
    def simulate_(self):
        # Initialize Monitors
        monitors = (lab.monitors.TemporalAverage(period=10.0))
        model, connectivity, coupling, integrator = self.tvb_model()
        # Initialize Simulator
        sim = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                      coupling=coupling, integrator=integrator,
                                      monitors=[monitors],
                                      conduction_speed=self.speed)
        sim.configure()
        SC = connectivity.weights
        # Run a warm up period to get rid of transitory dynamics
        FCloc = []
        tavg = []
        coeff = []
        # Step into the simulation
        #for i in range(0, 20):
        res = sim.run(simulation_length=self.sim_length)
        # Resulting time series data is stored as two arrays,
        # one with times and one with the signal values
        tavg_time = np.squeeze(res[0][0])
        tavg_data = np.squeeze(res[0][1])
        tavg.append(tavg_data)
        print(tavg)
        # Append the result of the functional connectivity per step
        FCloc.append(self.compFC(tavg_data))
        #self.plot_FCSC(SC, FCloc[-1], i)
        coeff.append(self.corrSCFC(SC, FCloc[-1]))
        print("Correlation = {}\n".format(coeff))
        return np.average(coeff)

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        bound_coupling = [0.0, 3.0]
        speed = [1.0, 2.0]
        return {'coupling': random.uniform(bound_coupling[0],
                                           bound_coupling[1]),
                'speed': random.uniform(speed[0], speed[1])}

    def bounding_func(self, individual):
        return individual

    def simulate(self, trajectory):
        self.id = trajectory.individual.ind_idx
        self.speed = trajectory.individual.speed
        self.coupling = trajectory.individual.coupling

        self.sim_length = 100

        # Start simulation
        fitness = self.simulate_()
        # Return the last correlation coeficient as fitness of the model
        return fitness


def end(self):
    logger.info("End of all experiments. Cleaning up...")
    # There's nothing to clean up though


def main():
    from ltl import DummyTrajectory
    from ltl.paths import Paths

    # TODO: Set root_dir_path here
    paths = Paths('pse', dict(run_num='test'), root_dir_path='.')

    fake_traj = DummyTrajectory()
    optimizee = TVBOptimizee(fake_traj, 0)
    fake_traj.individual = sdict(optimizee.create_individual())

    testing_error = optimizee.simulate(fake_traj)
    print("Testing error is %s", testing_error)


if __name__ == "__main__":
    main()
