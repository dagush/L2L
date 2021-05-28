import argparse
import pyNetLogo


parser = argparse.ArgumentParser()
parser.add_argument('--netlogo_home', type=str, help='Path to NetLogo')
parser.add_argument('--netlogo_version', help='NetLogo Version')
parser.add_argument('--model', type=str, help='Path to model')
parser.add_argument('--ticks', type=int, help='Simulation steps')
parser.add_argument('--individual_no', type=int, help='Individual Number')
args = parser.parse_args()
print(args)


def run_model():
    netlogo = pyNetLogo.NetLogoLink(
        gui=False, netlogo_home=netlogo_home, netlogo_version=netlogo_version)
    netlogo.load_model(model)
    netlogo.command('set simulation_index {}'.format(individual_no))
    netlogo.command('setup')
    netlogo.command('set simulation_index {}'.format(individual_no))
    netlogo.repeat_command('go', ticks)

    fitness = netlogo.repeat_report(['fitness_value'], 1, go='go')
    print(fitness.iloc[-2].values)
    netlogo.kill_workspace()
    return fitness


netlogo_home = args.netlogo_home
netlogo_version = str(args.netlogo_version)
model = args.model
ticks = args.ticks
individual_no = args.individual_no
fit = run_model()
