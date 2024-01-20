from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os, sys, time, shutil, pathlib, shutil
import numpy as np
from numpy.linalg import norm       # to get the norm of a vector
import matplotlib.pyplot as plt

from utils.tools import is_close_to, display_joint_properties, test_training, sample_line, sample_traj4pts
from utils.tools import welcome, plot_test, moving_average, get_files_by_date

from utils.RoboticArm_2DOF import RoboticArm_2DOF_PyBullet

from drl_training import env, DT, EPSILON

# for directory in models/ ad starting by 2DOF
for directory in [d for d in os.listdir('models/') if d.startswith('2DOF')]:
    training_dir = pathlib.Path('models') / directory
    list_files = [f for f in get_files_by_date(training_dir/'ZIP') if f.startswith('rl_model')]

    err_mean  = np.inf
    error_mean = []
    max_steps = 256
    env._max_episode_steps = None

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
        error_mean.append(e_mean)
        print(f"\t e_mean: {e_mean*100:6.2f}, e_std: {e_std*100:6.2f} cm")
        if e_mean < err_mean: 
            best_train = file
            err_mean   = e_mean
                
    print(f"Best train: {best_train:30s}, error: {err_mean*100:.2f} cm")


    plt.figure(figsize=(10, 6))
    plt.bar(range(len(list_files)), error_mean)
    plt.xticks(range(len(list_files)), list_files, rotation=90)
    plt.xlabel('Training Files')
    plt.ylabel('Mean Error')
    plt.title('Mean Error for Each Training File')
    plt.show()

    min_error_index = np.argmin(error_mean)
    best_train = list_files[min_error_index]
    rank = min_error_index + 1
    print(f"The zip training file with the smallest error is {best_train} (Rank: {rank})")

    print(f"The zip training file with the smallest error is {best_train} (Rank: {rank})")
    print(f"The smallest mean error is {err_mean*100:.2f} cm")
