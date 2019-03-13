#!/usr/bin/env bash

python3 -m baselines.ddpg.main --env-id RoboschoolHalfCheetah-v1 --logfile HalfCheetah_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHalfCheetah-v1 --logfile HalfCheetah_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHalfCheetah-v1 --logfile HalfCheetah_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHalfCheetah-v1 --logfile HalfCheetah_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHalfCheetah-v1 --logfile HalfCheetah_05.dat &

python3 -m baselines.ddpg.main --env-id RoboschoolHopper-v1 --logfile Hopper_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHopper-v1 --logfile Hopper_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHopper-v1 --logfile Hopper_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHopper-v1 --logfile Hopper_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolHopper-v1 --logfile Hopper_05.dat &

wait

python3 -m baselines.ddpg.main --env-id RoboschoolInvertedPendulum-v1 --logfile InvertedPendulum_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedPendulum-v1 --logfile InvertedPendulum_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedPendulum-v1 --logfile InvertedPendulum_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedPendulum-v1 --logfile InvertedPendulum_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedPendulum-v1 --logfile InvertedPendulum_05.dat &

python3 -m baselines.ddpg.main --env-id RoboschoolInvertedDoublePendulum-v1 --logfile InvertedDoublePendulum_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedDoublePendulum-v1 --logfile InvertedDoublePendulum_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedDoublePendulum-v1 --logfile InvertedDoublePendulum_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedDoublePendulum-v1 --logfile InvertedDoublePendulum_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolInvertedDoublePendulum-v1 --logfile InvertedDoublePendulum_05.dat &

wait

python3 -m baselines.ddpg.main --env-id RoboschoolAnt-v1 --logfile Ant_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolAnt-v1 --logfile Ant_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolAnt-v1 --logfile Ant_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolAnt-v1 --logfile Ant_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolAnt-v1 --logfile Ant_05.dat &

python3 -m baselines.ddpg.main --env-id RoboschoolReacher-v1 --logfile Reacher_01.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolReacher-v1 --logfile Reacher_02.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolReacher-v1 --logfile Reacher_03.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolReacher-v1 --logfile Reacher_04.dat &
python3 -m baselines.ddpg.main --env-id RoboschoolReacher-v1 --logfile Reacher_05.dat &

wait