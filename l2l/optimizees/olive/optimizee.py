import logging
from enum import Enum
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as pl

#from singleneuron_class import SingleNeuron

from l2l import sdict
from l2l.optimizees.optimizee import Optimizee
import ast

import eden_tools

from pyneuroml import pynml
from pyneuroml.pynml import print_comment_v
from pyneuroml.lems import LEMSSimulation
import neuroml as nml
import neuroml.writers as writers
import neuroml.loaders as loaders
import neuroml.utils as utils
from neuroml.nml.nml import parse as nmlparse
logger = logging.getLogger("ltl-neuron")

import scipy.fftpack
from scipy.signal import find_peaks
from scipy import integrate

from collections import namedtuple

NeuronOptimizeeParameters = namedtuple('NeuronOptimizeeParameters', [])

class SingleNeuronFit:
    def __init__(self, neuron_name, 
                 na_s_soma = 30, kdr_soma = 30, k_soma = 15, cal_soma = 20, cah_dend = 10, 
                 kca_dend = 35, h_dend = 25, na_axon = 200, k_axon = 200, leak = 1/3e-2,
                 task=1, simulator="Eden", path="/home/jovyan/work/ClassNora/L2L/l2l/optimizees/olive", targets = [20, 80, 0, -55]):
        
        random.seed(12345)
        
        # inputs
        self.neuron_name = neuron_name
        self.wd_path = os.getcwd()
        self.path = path
        self.task = task
        self.simulator = simulator
        self.targets = targets
        
        if self.task == 2:
            self.long_pulse_blockIds = []
            os.chdir(self.wd_path)
            rawdata_path = os.path.realpath("../dataNora")
            self.experimental_data = SingleNeuron(self.neuron_name, path = rawdata_path)
        
        #parameters and variables experimental part
        self.experimental_temp = 26
        if self.task == 2:
            self.experimental_current_signals = []
            self.experimental_voltage_signals = {}
            self.experimental_time_axis = []
        
            #pulse characteristics
            self.t_pulse_start = 0
            self.t_end = 0
            self.pulse_duration = 0
            self.pulse_heights = []

        #parameters and variables model part
        self.dt = 0.05
        self.model_init_time = 5000 #ms
        if self.task == 1:
            self.run_time = 5000 #ms
        self.ind_break = int(self.model_init_time/self.dt)+1
        self.LEMS_filename = " "
        self.results_Eden = {}
        self.results_Neuron = {}
        self.results = {}
        self.model_current_signals = []
        self.model_voltage_signals = {}
        self.model_time_axis = []
               
        #parameters and variables optimizer part
        self.fitness = -1000
        
        #functions experimental part
        if self.task == 2:
            self.get_long_pulse_blockIds()
            self.get_experimental_current_signals()
            self.get_experimental_voltage_signals()
            self.get_pulse_characteristics()
            
        #functions model part
        self.change_condDensity(na_s_soma, "na_s_soma")
        self.change_condDensity(kdr_soma, "kdr_soma")
        self.change_condDensity(k_soma, "k_soma")
        self.change_condDensity(cal_soma, "cal_soma")
        self.change_condDensity(cah_dend, "cah_dend")
        self.change_condDensity(kca_dend, "kca_dend")
        self.change_condDensity(h_dend, "h_dend")
        self.change_condDensity(na_axon, "na_axon")
        self.change_condDensity(k_axon, "k_axon")
        self.change_condDensity(leak, "leak")
        self.create_NML_network()
        
        if self.simulator == "Eden":
            self.run_network_with_Eden()
            self.results = self.results_Eden.copy()
        elif self.simulator == "Neuron":
            self.run_network_with_Neuron()
            self.results = self.results_Neuron.copy()

        if self.task == 2:
            self.split_model_data_in_blocks()
            
        #functions optimizee part
        self.compute_fitness()

    def get_long_pulse_blockIds(self):
        blocks_IV = []
        for i in range(len(self.experimental_data.get_blocknames(printing='off'))):
            if '-IV' in self.experimental_data.blocks[i].file_origin:
                blocks_IV.append(i)
        self.long_pulse_blockIds = blocks_IV

    def get_experimental_current_signals(self):
        # obtain current traces (channel_index 1) and time axis from experiment with blockId 0
        self.experimental_current_signals = self.experimental_data.blocks[self.long_pulse_blockIds[0]].channel_indexes[
            1].analogsignals

        time_axis_unit = 'ms'
        time_axis = self.experimental_current_signals[0].times
        self.experimental_time_axis = time_axis.rescale(time_axis_unit)

    def get_experimental_voltage_signals(self):
        for blockId in self.long_pulse_blockIds:
            self.experimental_voltage_signals[blockId] = self.experimental_data.blocks[blockId].channel_indexes[
                0].analogsignals

    def get_pulse_characteristics(self):
        # determine pulse start and end time
        t_pulse_start = 0
        flat_current_signal = np.squeeze(
            np.array(self.experimental_current_signals[0]))  # consider only the first pulse
        for i in range(1, len(flat_current_signal)):
            if abs(flat_current_signal[i] - flat_current_signal[i - 1]) > 10:
                if t_pulse_start == 0:
                    t_pulse_start = int(round(self.experimental_time_axis[i - 1]))
                    ind_pulse_start = i - 1
                else:
                    t_pulse_end = int(round(self.experimental_time_axis[i - 1]))
                    ind_pulse_end = i - 1
        if t_pulse_start == 0:
            print('no height step detected')
            return
        else:
            self.pulse_duration = t_pulse_end - t_pulse_start
        self.t_pulse_start = t_pulse_start
        self.t_end = int(round(self.experimental_time_axis[-1]))

        # determine pulse heights
        pulse_heights = []  # and save in a list
        margin = 100  # compute average height over whole width of pulse -margin at sides
        for current_signal in self.experimental_current_signals:
            flat_current_signal = np.squeeze(np.array(current_signal))
            av_height = sum(flat_current_signal[ind_pulse_start + margin:ind_pulse_end - margin]) / (
                        ind_pulse_end - ind_pulse_start - 2 * margin)
            pulse_heights.append(av_height)
        self.pulse_heights = pulse_heights

    def create_NML_network(self):
        os.chdir(self.path)

        # Create NeuroML file
        nml_doc = nml.NeuroMLDocument(id="net")

        # Include cell file
        
        incl = nml.IncludeType(href="C" + self.neuron_name[2:] + "_scaled_resample_5.cell.nml")
        nml_doc.includes.append(incl)

        # Create network
        net = nml.Network(id="net", type="networkWithTemperature", temperature=str(self.experimental_temp) + "degC")
        nml_doc.networks.append(net)

        # Create population
        comp_id = "C" + self.neuron_name[2:]
        pop = nml.Population(id="pop", component=comp_id, type="populationList", size="1")
        net.populations.append(pop)

        loc = nml.Location(x="0", y="0", z="0")

        inst = nml.Instance(id="0", location=loc)
        pop.instances.append(inst)

        # Create pulse generator
        if self.task == 2:
            for pulse_nr in range(len(self.pulse_heights)):
                p_delay = self.t_pulse_start + pulse_nr * self.t_end
                pg = nml.PulseGenerator(id="iclamp" + str(pulse_nr), delay=str(p_delay) + "ms",
                                        duration=str(self.pulse_duration) + "ms",
                                        amplitude=str(self.pulse_heights[pulse_nr]) + "pA")
                nml_doc.pulse_generators.append(pg)

                # Add pg to cell
                il = nml.InputList(id="clamps" + str(pulse_nr), component=pg.id, populations="pop")
                ip = nml.Input(id="0", target="../pop/0/" + comp_id, segmentId="0", destination="synapses")
                il.input.append(ip)
                net.input_lists.append(il)

        nml_file = self.neuron_name[2:] + ".net.nml"
        writers.NeuroMLWriter.write(nml_doc, nml_file)

        nml_file_dir = os.path.dirname(os.path.realpath(nml_file))

        # print("Written network file to: "+nml_file_dir+"/"+nml_file)

        sim_id = self.neuron_name[2:]
        if self.task == 1:
            sim_dur_ms = self.model_init_time + self.run_time
        elif self.task == 2:
            sim_dur_ms = len(self.pulse_heights) * self.t_end
        quantity = "pop/0/"+comp_id+"/0/v"
        target = 'net'

        ls = LEMSSimulation(sim_id, sim_dur_ms, self.dt, target=target)
        ls.include_neuroml2_file(nml_file)

        #disp0 = 'display0'
        #ls.create_display(disp0, "Spiking pattern", "-90", "50")
        #ls.add_line_to_display(disp0, 'v', quantity)

        of0 = 'Volts_file'
        ls.create_output_file(of0, "%s.v.dat" % sim_id)
        ls.add_column_to_output_file(of0, 'v', quantity)

        self.LEMS_filename = "%s_simulation.xml" % sim_id
        ls.save_to_file(file_name=self.LEMS_filename)

        os.chdir(self.wd_path)

    def run_network_with_Eden(self):
        os.chdir(self.path)
        self.results_Eden = eden_tools.runEden(self.LEMS_filename, verbose=True)
        os.chdir(self.wd_path)

    def run_network_with_Neuron(self):
        os.chdir(self.path)
        self.results_Neuron = pynml.run_lems_with_jneuroml_neuron(self.LEMS_filename, verbose=True, nogui=True)
        os.chdir(self.wd_path)

    def split_model_data_in_blocks(self):
        length = int(self.t_end / self.dt)
        mod_traces = {}

        # split time axis
        t_short = self.results_Neuron['t'][:length]  # Eden
        self.model_time_axis = [t * 1000 for t in t_short]

        # split voltage signals
        traces = self.results_Neuron['pop/0/C%s/0/v' % self.neuron_name[2:]]  # Eden
        for pulse_nr in range(len(self.pulse_heights)):
            temp = traces[pulse_nr * length:(pulse_nr + 1) * length]
            mod_traces[pulse_nr] = [volt * 1000 for volt in temp]
        self.model_voltage_signals = mod_traces

    def expId_from_blockId(self, blockId):
        return int(self.experimental_data.blocks[blockId].file_origin[-9:-7])

    def blockId_from_expId(self, expId):
        for i in self.long_pulse_blockIds:
            if expId < 10:
                if '0' + str(expId) + '-IV' in self.experimental_data.blocks[i].file_origin:
                    blockId = i
                    break
            else:
                if str(expId) + '-IV' in self.experimental_data.blocks[i].file_origin:
                    blockId = i
                    break
        return blockId

    def plot_long_pulse_all_experiments(self):
        nr_of_experiments = len(self.long_pulse_blockIds)
        fig, ax = plt.subplots(nrows=nr_of_experiments + 1, ncols=1, figsize=(8, 4 * nr_of_experiments))
        fig.tight_layout(h_pad=5)

        # plot experimental current signals
        for current_signal in self.experimental_current_signals:
            flat_current_signal = np.squeeze(np.array(current_signal))
            ax[0].plot(self.experimental_time_axis, flat_current_signal)
            ax[0].set_ylabel("1.0 pA")
            ax[0].set_xlabel("time in ms")
            ax[0].set_title("Experimental inputs", size=16)

        # plot experimental voltage signals
        window = 1
        for blockId in self.long_pulse_blockIds:
            for voltage_signal in self.experimental_voltage_signals[blockId]:
                flat_voltage_signal = np.squeeze(np.array(voltage_signal))
                ax[window].plot(self.experimental_time_axis, flat_voltage_signal)
            ax[window].set_ylabel("1.0 mV")
            ax[window].set_xlabel("time in ms")
            expId = self.expId_from_blockId(blockId)
            ax[window].set_title("Experiment number " + str(expId), size=16)
            window += 1

        fig.suptitle(str(self.neuron_name), fontsize=20, y=1.02)
        plt.show()

    def plot_long_pulse_model_and_selected_experiment(self, expId=None):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14, 12))
        fig.tight_layout(h_pad=5)

        # plot experimental current signals
        for current_signal in self.experimental_current_signals:
            flat_current_signal = np.squeeze(np.array(current_signal))
            ax[0].plot(self.experimental_time_axis, flat_current_signal)
        ax[0].set_ylabel("1.0 pA")
        ax[0].set_xlabel("time in ms")
        ax[0].set_title("Experimental inputs", size=16)

        # plot experimental voltage signals of chosen experiment
        if expId is not None:
            blockId = self.blockId_from_expId(expId)
        else:
            blockId = self.long_pulse_blockIds[0]
            expId = self.expId_from_blockId(blockId)

        for voltage_signal in self.experimental_voltage_signals[blockId]:
            flat_voltage_signal = np.squeeze(np.array(voltage_signal))
            ax[1].plot(self.experimental_time_axis, flat_voltage_signal)
        ax[1].set_ylabel("1.0 mV")
        ax[1].set_xlabel("time in ms")
        ax[1].set_title("Experimental voltage traces (experiment number " + str(expId) + ")", size=16)

        # plot model voltage signals
        for pulse_nr in range(len(self.pulse_heights)):
            ax[2].plot(self.model_time_axis, self.model_voltage_signals[pulse_nr])
        ax[2].set_ylabel("1.0 mV")
        ax[2].set_xlabel("time in ms")
        ax[2].set_title("Model voltage traces", size=16)

        fig.suptitle(str(self.neuron_name), fontsize=20, y=1.05)
        plt.show()

    def change_condDensity(self, value, channel):
        os.chdir(self.path)
        # na_s_soma, kdr_soma, k_soma, cal_soma, cah_dend, kca_dend, h_dend, na_axon, k_axon, leak
        leak_channels = ["leak_soma", "leak_axon", "leak_dend_prox", "leak_dend_dist"]
        if channel == "leak":
            for i in range(len(leak_channels)):
                 self.change_condDensity(value, leak_channels[i])
            return
        
        
        doc = nmlparse("C" + self.neuron_name[2:] + "_scaled_resample_5.cell.nml")
        channel_densities = doc.cells[0].biophysical_properties.membrane_properties.channel_densities
        nr_of_channels = len(channel_densities)
        for i in range(nr_of_channels):
            if channel_densities[i].id == channel:
                channel_densities[i].cond_density = str(value) + ' mS_per_cm2'
                print("succesfully changed parameter value")
                break
            if i == nr_of_channels:
                print("channel not found; value was not changed")
        writers.NeuroMLWriter.write(doc, "C" + self.neuron_name[2:] + "_scaled_resample_5.cell.nml")
        os.chdir(self.wd_path)

    def rerun_model(self):
        self.create_NML_network()
        if self.simulator == "Eden":
            self.run_network_with_Eden()
            self.results = self.results_Eden.copy()
        elif self.simulator == "Neuron":
            self.run_network_with_Neuron()
            self.results = self.results_Neuron.copy()
        self.split_model_data_in_blocks()
       
    def compute_fitness(self):
        if self.task == 1:
            #STO amplitude
            data_full = self.results['pop/0/C'+self.neuron_name[2:]+'/0/v']
            time_full = self.results['t']
            data_inV = data_full[self.ind_break:]
            data = [data_inV[i] * 1000 for i in range(len(data_inV))]
            time = time_full[self.ind_break:]
            
            top_peaks, _ = find_peaks(data, prominence = 8)
            bottom_peaks, _ = find_peaks([-data[i] for i in range(len(data))])
            
            max_val = np.mean([data[i] for i in top_peaks])
            min_val = np.mean([data[i] for i in bottom_peaks])
            mean_val = (max_val + min_val)/2
            ampl = max_val - min_val
            
            #check if everything went right
            plt.figure()
            plt.plot(time, data)
            plt.plot([self.model_init_time*0.001, (self.model_init_time+self.run_time)*0.001], [max_val, max_val], 'r', label = 'maximum')
            plt.plot([self.model_init_time*0.001, (self.model_init_time+self.run_time)*0.001], [min_val, min_val], 'm', label = 'minimum')
            for i in top_peaks:
                plt.plot(time[i], data[i], 'xr')
            for i in bottom_peaks:
                plt.plot(time[i], data[i], 'xr')
            plt.xlabel("time (s)")
            plt.ylabel("membrane potential (mV)")
            plt.title('C'+self.neuron_name[2:]+", verify that peaks, the minimum and the maximum were determined correctly", fontsize=16)
            plt.legend(loc='center right')
            plt.savefig('peak_detection.png')            
                         
            #STO frequency
            if len(top_peaks) > 2:
                freq = (len(top_peaks) - 1)/(time[top_peaks[-1]] - time[top_peaks[0]])
            else:
                freq = 0
            
            #STO symmetry
            data_centered = data - mean_val
            data_abs = abs(data_centered)
            data_int = integrate.cumtrapz(data_centered[top_peaks[0]:top_peaks[-1]], time[top_peaks[0]:top_peaks[-1]], initial=0)
            surface = integrate.cumtrapz(data_abs[top_peaks[0]:top_peaks[-1]], time[top_peaks[0]:top_peaks[-1]], initial=0)
            symm = data_int[-1]/surface[-1]
            
            #Compute finess
            fitness_ampl = - ((self.targets[0] - ampl) / self.targets[0])**2
            fitness_freq = - ((self.targets[1] - freq) / self.targets[1])**2
            fitness_symm = - (self.targets[2] - symm)**2
            fitness_mean = - ((self.targets[3] - mean_val) / self.targets[3])**2
                    
            self.fitness = fitness_ampl + fitness_freq + fitness_symm + fitness_mean


class NeuronOptimizee(Optimizee):

    def __init__(self, trajectory, seed):
        super(NeuronOptimizee, self).__init__(trajectory)

        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)
        
    def init_(self, trajectory):
        return

    def simulate_(self):
        self.neuron_name = '20160802D'
        self.snf = SingleNeuronFit(self.neuron_name, self.na_s_soma, self.kdr_soma, self.k_soma, self.cal_soma, self.cah_dend,
                                                         self.kca_dend, self.h_dend, self.na_axon, self.k_axon, self.leak)
        return self.get_fitness()

    def get_fitness(self):
        return self.snf.fitness
    
    def simulate(self, trajectory):

        self.id = trajectory.individual.ind_idx
        self.na_s_soma = trajectory.individual.na_s_soma
        self.kdr_soma = trajectory.individual.kdr_soma
        self.k_soma = trajectory.individual.k_soma
        self.cal_soma = trajectory.individual.cal_soma
        self.cah_dend = trajectory.individual.cah_dend
        self.kca_dend = trajectory.individual.kca_dend
        self.h_dend = trajectory.individual.h_dend
        self.na_axon = trajectory.individual.na_axon
        self.k_axon = trajectory.individual.k_axon
        self.leak = trajectory.individual.leak
        self.init_(trajectory)

        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.labelsize'] = 14

        self.simulate_()

        print(self.get_fitness())
        return (self.get_fitness())

    # Returned by the prev fading mem test. return readout_delay, testing_perf_mean

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        # na_s_soma, kdr_soma, k_soma, cal_soma, cah_dend, kca_dend, h_dend, leak_all
        self.bound_na_s_soma = [15, 60]
        self.bound_kdr_soma = [15, 60]
        self.bound_k_soma = [7.5, 30]
        self.bound_cal_soma = [10, 40]
        self.bound_cah_dend = [5, 20]
        self.bound_kca_dend = [17.5, 70]
        self.bound_h_dend = [12.5, 50]
        self.bound_na_axon = [100, 400]
        self.bound_k_axon = [100, 400]
        self.bound_leak = [0.65e-2, 2.6e-2]
                 
        return {'na_s_soma': self.random_state.rand() * (self.bound_na_s_soma[1] - self.bound_na_s_soma[0]) + self.bound_na_s_soma[0],
                'kdr_soma': self.random_state.rand() * (self.bound_kdr_soma[1] - self.bound_kdr_soma[0]) + self.bound_kdr_soma[0], 
                'k_soma': self.random_state.rand() * (self.bound_k_soma[1] - self.bound_k_soma[0]) + self.bound_k_soma[0],
                'cal_soma': self.random_state.rand() * (self.bound_cal_soma[1] - self.bound_cal_soma[0]) + self.bound_cal_soma[0], 
                'cah_dend': self.random_state.rand() * (self.bound_cah_dend[1] - self.bound_cah_dend[0]) + self.bound_cah_dend[0],
                'kca_dend': self.random_state.rand() * (self.bound_kca_dend[1] - self.bound_kca_dend[0]) + self.bound_kca_dend[0],
                'h_dend': self.random_state.rand() * (self.bound_h_dend[1] - self.bound_h_dend[0]) + self.bound_h_dend[0],
                'na_axon': self.random_state.rand() * (self.bound_na_axon[1] - self.bound_na_axon[0]) + self.bound_na_axon[0],
                'k_axon': self.random_state.rand() * (self.bound_k_axon[1] - self.bound_k_axon[0]) + self.bound_k_axon[0],
                'leak': self.random_state.rand() * (self.bound_leak[1] - self.bound_leak[0]) + self.bound_leak[0]}

    def bounding_func(self, individual):
        return individual

    def end(self):
        logger.info("End of all experiments. Cleaning up...")
        # There's nothing to clean up though


def main():
    from l2l.utils.environment import Environment
    from l2l.utils.experiment import Experiment
    from l2l import DummyTrajectory
    experiment = Experiment(root_dir_path='../../results')
    jube_params = {}
    try:
        trajectory, _ = experiment.prepare_experiment(
            name='test_trajectory',
            log_stdout=True,
            add_time=True,
            automatic_storing=True,
            jube_parameter=jube_params)
    except FileNotFoundError as fe:
        self.fail(
            "{} \n L2L is not well configured. Missing path file.".format(
                fe))
    paths = experiment.paths

    fake_traj = DummyTrajectory()
    optimizee = NeuronOptimizee(fake_traj, 0)
    fake_traj.individual = sdict(optimizee.create_individual())
    testing_error = optimizee.simulate(fake_traj)
    logger.info("Testing error is %s", testing_error)


if __name__ == "__main__":
    main()
