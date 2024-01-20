###########################################################################
# Copyright 2022-2023 Jean-Luc CHARLES
# Created: 2022-07-29
# version: 1.2 - 3 Dec 2023
# License: GNU GPL-3.0-or-later
###########################################################################

import os, sys, time, shutil, yaml, pathlib, shutil
import pybullet as p
import pybullet_data
import numpy as np
from numpy.linalg import norm       # to get the norm of a vector
from numpy import pi
import matplotlib.pyplot as plt

from utils.tools import is_close_to, display_joint_properties, test_training, sample_line, sample_traj4pts
from utils.tools import welcome, plot_test, moving_average, get_files_by_date

from utils.RoboticArm_2DOF import RoboticArm_2DOF_PyBullet

# the PyBullect connection:
pc = None

ROBOT  = "./urdf/RoboticArm_2DOF_2.urdf"
TARGET = "./urdf/target.urdf"

if 'env' in dir(): 
    env.close()
    del env

env = RoboticArm_2DOF_PyBullet(robot_urdf=ROBOT,      # mandatory
                               target_urdf=TARGET,    # mandatory
                               dt=1/240,              # mandatory, time step of the simulation [s] (~4ms) 
                               headless=False,        # to get the PyBullet graphical interface 
                               verbose=2)


DT      = 1/240          # the simulation time step
EPSILON = 1e-3           # the distance threshold betwwen the end effector and the target
SEED    = 132        # the seed for the random generators
MAX_EPISODE_STEPS = 256

URDF   = "./urdf/RoboticArm_2DOF_2.urdf"
TARGET = "./urdf/target.urdf"

if 'env' in dir():
    try:
        env.close()
    except:    
        del env

env    = RoboticArm_2DOF_PyBullet(robot_urdf  = URDF, 
                                  target_urdf = TARGET, 
                                  dt = DT,
                                  init_robot_angles = (113, -140),
                                  init_target_pos = (0.5, 0, 0.5),
                                  reward = 'reward_1',
                                  seed = SEED,
                                  epsilon = EPSILON,
                                  headless = True,  # no more graphical rendering for this round
                                  max_episode_steps = MAX_EPISODE_STEPS,
                                  verbose = 0)

policy = 'MlpPolicy'
tot_steps  = 5000000     # will take a few hours... (~ 4h on a core_I7 laptop) | value = 5000000
save_freq  = 100000      # save the networks weights every 'save_freq' steps | value = 100000
nb_steps   = 2048        # The number of steps to run per update (the size of the rollout buffer) | value = 2048
nb_epochs  = 10          # number of training iterations with the same dataset | value = 10
batch_size = 512         # size of the batch to train the network | value = 512
headless   = True        # no graphical renering for this round | value = True

experiment_time = time.localtime()
experiment_id = "_".join(['2DOF_RobotArm_PyBullet', 'PPO', time.strftime("%y-%m-%d_%H-%M-%S", experiment_time)])

training_dir = pathlib.Path('models') / experiment_id
training_dir.mkdir(parents=True, exist_ok=True)

# copy precious files in experiment_dir
for f in ('./2-DRL_training.ipynb', 'rewards.py', 'drl_training.py', 'utils/tools.py', './utils/RoboticArm_2DOF.py', 
             './utils/perf.py', './urdf/RoboticArm_2DOF_2.urdf'):
    base = os.path.basename(f)
    shutil.copyfile(f, training_dir / base)
print(f"Files copied in <{training_dir}>")
print(f"Training in directory <{training_dir}>")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

agent = PPO(policy, 
            env, 
            n_epochs = nb_epochs,
            n_steps = nb_steps,
            batch_size = batch_size,
            use_sde = False,
            seed = SEED,
            tensorboard_log = training_dir,
            verbose = 0)

checkpoint_callback = CheckpointCallback(save_freq = save_freq, 
                                         save_path = training_dir / 'ZIP')

# train agent
t0 = time.time()

agent.learn(total_timesteps = tot_steps, 
            callback = checkpoint_callback)
    
t = int(time.time()-t0)
h = int(t//3600)
m = int((t - h*3600)//60)
print(f"Training elapsed time: {h:02d}h {m:02d}m")

env.close()

print(f"Processing ZIP files in <{training_dir}>")

if 'env' in  dir():
    try:
        env.close()
    except:    
        del env
      
ROBOT  = "./urdf/RoboticArm_2DOF_2.urdf"
TARGET = "./urdf/target.urdf"

env    = RoboticArm_2DOF_PyBullet(robot_urdf  = ROBOT, 
                                  target_urdf = TARGET, 
                                  dt = DT,
                                  init_robot_angles = (113, -140),
                                  init_target_pos = (0.5, 0, 0.5),
                                  reward = 'reward_1',
                                  seed = SEED,
                                  epsilon = EPSILON,
                                  headless = True, 
                                  max_episode_steps = None,
                                  verbose=0)

list_files = [f for f in get_files_by_date(training_dir/'ZIP') if f.startswith('rl_model')]

from stable_baselines3 import PPO

err_mean  = np.inf
err_std   = np.inf
error     = []
max_steps = 256
env._max_episode_steps = None

print(len(list_files))
for i, file in enumerate(list_files):    
    print(f"{file:30s}", end="")
    err = []
    
    agent  = PPO.load(training_dir / 'ZIP' / file)
    obs, _ = env.reset(options={"dt": DT, 
                                "target_initial_pos": (0.5,0,0),
                                "robot_initial_angle_deg": (113, -140),
                                "randomize": False,
                                "epsilon": EPSILON})    
    
    for target_pos in ((0.5,0.,0.02), (1,0,0.5), (0.5,0,1), (0,0,0.5), (0.5,0,0.02)):
        env.set_target_position(np.array(target_pos))
        terminated, truncated, step_count, rewards, actions = False, False, 0, [], []
        while step_count < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated: break   
                
        dist_effect_target = norm(np.array(env.effector_pos) - target_pos)
        err.append(dist_effect_target)
        
    e_mean = np.array(err).mean()
    e_std  = np.array(err).std()
    error.append(err)
    print(f"\t e_mean: {e_mean*100:6.2f}, e_std: {e_std*100:6.2f} cm")
    if e_mean < err_mean: 
        best_train = file
        err_mean   = e_mean
        err_std    = e_std
        # error = np.array(error)            
        print(f"Best train: {best_train:30s}, error: {err_mean*100:.2f} cm")
    # print(f"Best train: {file:30s}, error: {e_mean*100:.2f} cm")