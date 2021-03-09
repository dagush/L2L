import logging
from enum import Enum
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as pl

from singleneuron_class import SingleNeuron

from l2l import sdict
from l2l.optimizees.optimizee import Optimizee
import ast

from pyneuroml import pynml
from pyneuroml.pynml import print_comment_v
from pyneuroml.lems import LEMSSimulation
import neuroml as nml
import neuroml.writers as writers
import neuroml.loaders as loaders
import neuroml.utils as utils
from neuroml.nml.nml import parse as nmlparse
logger = logging.getLogger("ltl-neuron")


class SingleNeuronFit:
    def __init__(self, neuron_name, v1, v2, path=""):
        random.seed(12345)


        # import eden_tools
        self.neuron_name = neuron_name
        self.path = path
        self.long_pulse_blockIds = []
        os.chdir(self.path)

        rawdata_path = os.path.realpath("../dataNora")
        self.experimental_data = SingleNeuron(self.neuron_name, path=rawdata_path)
        os.chdir(self.path)

        # experimental part
        self.experimental_temp = 26
        self.experimental_current_signals = []
        self.experimental_voltage_signals = {}
        self.experimental_time_axis = []

        # pulse characteristics
        self.t_pulse_start = 0
        self.t_end = 0
        self.pulse_duration = 0
        self.pulse_heights = []

        # model part
        self.dt = 0.05
        self.LEMS_filename = " "
        self.results_Eden = {}
        self.results_Neuron = {}
        self.model_current_signals = []
        self.model_voltage_signals = {}
        self.model_time_axis = []

        self.get_long_pulse_blockIds()
        self.get_experimental_current_signals()
        self.get_experimental_voltage_signals()
        self.get_pulse_characteristics()

        # self.get_experimental_temp()
        self.create_NML_network()
        # self.run_network_with_Eden()
        self.run_network_with_Neuron()
        self.split_model_data_in_blocks()

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

    def get_experimental_temp(self):
        temp_min = self.experimental_data.recording_metadata.bath_temp_min.iloc[0]
        temp_max = self.experimental_data.recording_metadata.bath_temp_max.iloc[0]
        self.experimental_temp = (temp_min + temp_max) / 2

    def create_NML_network(self):
        os.chdir(os.path.realpath('../NMLfiles'))

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
        sim_dur_ms = len(self.pulse_heights) * self.t_end
        quantity = "pop/0/" + comp_id + "/0/v"
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

        os.chdir(self.path)

    def run_network_with_Eden(self):
        os.chdir(os.path.realpath('../NMLfiles'))
        self.results_Eden = eden_tools.runEden(self.LEMS_filename, verbose=True)
        os.chdir(self.path)

    def run_network_with_Neuron(self):
        os.chdir(os.path.realpath('../NMLfiles'))
        self.results_Neuron = pynml.run_lems_with_jneuroml_neuron(self.LEMS_filename, verbose=True, nogui=True)
        os.chdir(self.path)

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
        os.chdir(os.path.realpath('../NMLfiles'))
        # na_s_soma, kdr_soma, k_soma, cal_soma, cah_dend, kca_dend, h_dend, leak_all
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
        os.chdir(default_path)

    def rerun_model(self):
        self.create_NML_network()
        self.run_network_with_Eden()
        self.split_model_data_in_blocks()


class NeuronOptimizee(Optimizee):

    def __init__(self, trajectory, seed):
        super(NeuronOptimizee, self).__init__(trajectory)

        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

    def init_(self, trajectory):
        return

    def simulate_(self):
        neuron_names_selection = ['20160802A', '20160802D']
        neuron_names_selection = ['20160802D']

        self.cellData = {}

        for neuron_name in neuron_names_selection:
            self.cellData[neuron_name] = SingleNeuronFit(neuron_name, self.v1, self.v2, self.path)
        return self.get_fitness()

    def get_fitness(self):
        return np.random.randint(5, size=1)

    def simulate(self, trajectory):

        self.id = trajectory.individual.ind_idx
        self.v1 = trajectory.individual.v1
        self.v2 = trajectory.individual.v2
        self.init_(trajectory)

        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.labelsize'] = 14

        self.path = os.getcwd()
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
        self.bound_gr = [1,2]
        self.na_s_soma = [0, 0]
        self.kdr_soma = [0, 0]
        self.k_soma = [0, 0]
        self.cal_soma = [0, 0]
        self.cah_dend = [0, 0]
        self.kca_dend = [0, 0]
        self.h_dend = [0, 0]
        self.leak_all = [0, 0]
        return {'v1': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0],
                'v2': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0]}

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
