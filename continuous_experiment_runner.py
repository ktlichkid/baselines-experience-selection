import subprocess
from time import sleep


import json
import numpy as np
from experience_selection.helpers.data_dirs import completed_experiments

RUN_BASE = 'python3 ./baselines/ddpg/experience_selection_main.py '

def stop_request():
    return False

def get_experiments():
    with open('experiments_to_run.json') as f:
        experiment_list = json.load(f)
    return experiment_list


def get_result_counts(experiments):
    return [completed_experiments(e) for e in experiments]

def execution_string(experiment):
    exstr = RUN_BASE
    for setting in experiment:
        exstr += '--{:s} {:s} '.format(setting, str(experiment[setting]))
    return exstr

def next_experiment():
    experiments = get_experiments()
    result_counts = get_result_counts(experiments)
    return experiments[np.argmin(result_counts)]

if __name__ == '__main__':

    while True:
        if not stop_request():
            experiment = next_experiment()
            exec_str = execution_string(experiment)
            print(exec_str)
            proc = subprocess.run(exec_str, shell=True)
            print(proc.stderr)
        sleep(10)