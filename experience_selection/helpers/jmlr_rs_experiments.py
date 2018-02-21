import sys
import json

possible_env_arguments = {
    '-ipsu': 'RoboschoolInvertedPendulumSwingup-v1',
    '-ip': 'RoboschoolInvertedPendulum-v1',
    '-reacher': 'RoboschoolReacher-v1',
    '-idp': 'RoboschoolInvertedDoublePendulum-v1',
    '-ant': 'RoboschoolAnt-v1',
    '-hopper': 'RoboschoolHopper-v1',
    '-cheetah': 'RoboschoolHalfCheetah-v1',
    '-walker': 'RoboschoolWalker2d-v1'
}

if __name__ == '__main__':
    args = sys.argv[1:]

    experiments = []
    environments = []

    for arg in args:
        environments.append(possible_env_arguments[arg])
    if len(environments) == 0:
        environments = ['RoboschoolInvertedDoublePendulum-v1',
                        'RoboschoolReacher-v1',
                        'RoboschoolHopper-v1',
                        'RoboschoolHalfCheetah-v1',
                        ]

    base_settings = {
        'project_dir': 'jmlr_es_rs',
        'noise-type': 'adaptive-param_0.2',
        'buffer_sample': 'uniform',
    }

    for env in environments:
        for db_sze, db_sze_str in zip([1e6, 2e5, 4e4], ['1m', '200k', '40k']):
            for ows in ['FIFO', 'expl_1.2', 'tde_1.2', 'resv']:
                if ows == 'FIFO' or db_sze < 1e6:
                    setting = base_settings.copy()
                    setting['env-id'] = env
                    setting['buffer_overwrite'] = ows
                    setting['buffer_size'] = db_sze
                    setting['experiment_name'] = '{:s}_{:s}'.format(db_sze_str, ows)
                    experiments.append(setting)

    with open('../../experiments_to_run.json', mode='w') as f:
        json.dump(experiments,f,indent=4)