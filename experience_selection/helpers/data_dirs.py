import os
import json
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError


def get_and_prepare_log_dir(experiment_def):
    log_dir = next_save_dir(experiment_def)
    record_settings_and_assert_correct(experiment_def)
    clean_experiment_def(experiment_def)
    return log_dir


def clean_experiment_def(experiment_def):
    del experiment_def['project_dir']
    del experiment_def['experiment_name']


def full_experiment_dir(experiment_def):
    base = os.environ['RESULTS_BASE_DIR']
    eid = experiment_def.get('env-id')
    if eid is None:
        eid = experiment_def['env_id']
    return '{:s}/{:s}/{:s}/{:s}'.format(
        base,
        experiment_def['project_dir'],
        eid,
        experiment_def['experiment_name'])


def completed_experiments(experiment_def):
    try:
        experiment_dir = full_experiment_dir(experiment_def)
        results = len(tf.gfile.Glob('{:s}/run_*'.format(experiment_dir)))
    except NotFoundError:
        results = 0
    return results


def next_save_dir(exeriment_def):
    idx = completed_experiments(exeriment_def)
    save_dir = '{:s}/run_{:03d}'.format(
        full_experiment_dir(exeriment_def), idx)
    tf.gfile.MakeDirs(save_dir)
    return save_dir


def record_settings_and_assert_correct(experiment_def):
    settings_file_name = '{:s}/experiment_definition.json'.format(
        full_experiment_dir(experiment_def))
    try:
        with open(settings_file_name, mode='r') as f:
            prev_set = json.load(f)
            if not prev_set == experiment_def:
                for setting_name in prev_set:
                    if not prev_set[setting_name] == experiment_def[setting_name]:
                        print('Settings do not match with previously performed experiments with '
                              'the same name:')
                        print('previous setting:')
                        print(prev_set[setting_name])
                        print('current setting:')
                        print(experiment_def[setting_name])
                        raise ValueError('Settings mismatch')

    except (json.decoder.JSONDecodeError, FileNotFoundError):
        with open(settings_file_name, mode='w') as f:
            json.dump(fp=f, obj=experiment_def, indent=4)
