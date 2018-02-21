import argparse
import time
import json
import os

from tensorflow.python.framework.errors_impl import NotFoundError

from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
#from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import roboschool
import gym
import tensorflow as tf
from mpi4py import MPI

from experience_selection.experience_buffer import ExperienceBuffer, ESMemoryAdapter


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    #memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
    memory = ESMemoryAdapter(
        limit=int(kwargs['buffer_size']),
        action_shape=env.action_space.shape,
        observation_shape=env.observation_space.shape,
        forgetting_factor=kwargs['gamma'],
        overwrite_policy=kwargs['buffer_overwrite'],
        sample_policy=kwargs['buffer_sample'],
        batch_size=kwargs['batch_size']
    )
    del kwargs['buffer_size']
    del kwargs['buffer_overwrite']
    del kwargs['buffer_sample']


    # observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='RoboschoolHalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--project_dir', type=str, default='test')
    parser.add_argument('--experiment_name', type=str, default='default') # results are saved in
    # /data/<project-dir>/<env-id>/<experiment-name>/<iteration>/
    boolean_flag(parser, 'evaluation', default=False)

    # experience slection settings ====================================
    parser.add_argument('--buffer_size', type=int, default=1e4)  # 'FIFO', 'expl_xx' (
    parser.add_argument('--buffer_overwrite', type=str, default='expl_1.2') # 'FIFO', 'expl_xx' (
    # stochastic exploration magnitude based with alpha = xx), tde_xx (stochastic TDE based with alpha = xx), 'resv' (Reservoir sampling)
    parser.add_argument('--buffer_sample', type=str, default='PER_0.7')
    #'uniform', 'PER_xx (TDE rank based with alpha = xx)



    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

def determine_logdir(args):
    base = './results/{:s}/{:s}/{:s}'.format(args['project_dir'], args['env_id'],
                                             args['experiment_name'])
    try:
        previous_results = len(tf.gfile.Glob('{:s}/run_*'.format(base)))
        idx = previous_results
    except NotFoundError:
        idx = 0
    fulldir = '{:s}/run_{:03d}'.format(base, idx)
    tf.gfile.MakeDirs(fulldir)
    settings_file_name = '{:s}/experiment_definition.json'.format(base)
    try:
        with open(settings_file_name, mode='r') as f:
            prev_set = json.load(f)
            assert prev_set == args, 'attempting to perform experiments with different settings ' \
                                     'in an already written to experiment folder'
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        with open(settings_file_name, mode='w') as f:
            json.dump(fp=f, obj=args, indent=4)

    del args['project_dir']
    del args['experiment_name']
    return fulldir

if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        dir = determine_logdir(args)
        logger.configure(dir=dir)
    # Run actual script.
    run(**args)
