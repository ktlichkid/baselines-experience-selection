import datetime
import glob
from os.path import expanduser
from time import sleep

import pandas as pd
import subprocess

from experiments.definitions.experiment_definitions_generic import experiment_definitions_generic as experiment_definitions
from interpretation.pandas_logger import PandaLogChecker

BASEDIR = '~/.tfdata/META/roboschool/'
FILENAME = 'experiment_variations.csv'
REQUIRED_FIELDS = ['base', 'name', 'results', 'execute', 'description']
# RUN_COMMAND_BASE = 'source ~/tensorflow3/bin/activate && export PYTHONPATH=$PYTHONPATH:$(
# pwd)/benchmarks/gym_torcs && python3 torcs_experiment.py'
RUN_COMMAND_BASE = 'python3 generic_roboschool_test.py'


class ExperimentQue(object):
    def __init__(self):
        self.df = pd.read_csv(expanduser(BASEDIR + FILENAME))
        self.log_checker = PandaLogChecker()
        self.run_continuously()

    @property
    def active_experiments(self):
        return self.df[self.df['execute'] == True]

    @property
    def next_experiment_to_perform(self):
        ae = self.active_experiments
        idx = ae['results'].idxmin()
        return self.df.iloc[idx]

    @classmethod
    def parameter_string(cls, exp, trial_idx):
        return_str = ' --experiment_definition "{:s}" --name "{:s}"'.format(
            exp['base'], exp['name'])
        for k, v in exp.dropna().items():
            if k not in REQUIRED_FIELDS:
                """
                if isinstance(v, bool):
                    if v:
                        return_str += ' --{:s}'.format(k)
                elif isinstance(v, str):
                """
                if isinstance(v, str):
                    return_str += ' --{:s} "{:s}"'.format(k, v)
                else:
                    return_str += ' --{:s} {:s}'.format(k, str(v))
        return return_str

    @classmethod
    def execution_string(cls, param_string):
        return RUN_COMMAND_BASE + param_string

    @property
    def next_execution_string(self):
        exp = self.next_experiment_to_perform
        trial = exp['results'] + 1
        return self.execution_string(self.parameter_string(exp, trial))

    @property
    def stop_request(self):
        potential_stop_files = glob.glob('stop*')
        if len(potential_stop_files) > 0:
            stop_date_time_vals = [int(s) for s in potential_stop_files[0].split('_') if
                                   s.isdigit()]
            if len(stop_date_time_vals) > 0:
                stop_date_time = datetime.datetime(*stop_date_time_vals)
                return stop_date_time < datetime.datetime.now()
            else:
                return True
        else:
            return False

    def update_run_counts(self):
        self.df = pd.read_csv(expanduser(BASEDIR + FILENAME))
        for idx in range(len(self.df)):
            exp = self.df.iloc[idx]
            expdef = experiment_definitions.get_settings_for_experiment_name(exp['base'])
            expdef['variation_name'] = exp['name']
            self.df.loc[idx, 'results'] = self.log_checker.get_trial_idx(expdef)
        self.df.to_csv(expanduser(BASEDIR + FILENAME + '_out.csv'))

    def run_continuously(self):
        while True:
            if not self.stop_request:
                self.update_run_counts()
                exec_str = self.next_execution_string
                print(exec_str)
                proc = subprocess.run(exec_str, shell=True)
                print(proc.stderr)
            sleep(10)


if __name__ == '__main__':
    runner = ExperimentQue()
