from stable_baselines3 import PPO
import numpy as np
from numpy.linalg import norm
import os

from utils.tools import get_files_by_date

def perf_trajectoire(env, model_dir, DT, max_steps_nb=500, verbose=0):
    """
    We browse the training files to select the one giving the smallest mean distance 
    between the robot end effector and the 5 target positions figuring a diamond.    
    """
    q1_q2    = (113, -140)
    epsilon  = 1.e-3
    err_mean, err_std  = np.inf, np.inf
    error = []

    list_files = [f for f in get_files_by_date(model_dir / 'ZIP') if f.startswith('rl_model')]
    env._max_episode_steps = None
    
    for i, file in enumerate(list_files):            
        print(f">>> {file:30s}", end="")
        res, err = [], []
        agent = PPO.load(model_dir / 'ZIP' / file)
        obs, _ = env.reset(options={"dt": DT, 
                                    "target_initial_pos": (0.5,0,0),
                                    "robot_initial_angle_deg": q1_q2,
                                    "randomize": False,
                                    "epsilon": epsilon})    
        
        for target_pos in ((0.5,0.,0.02), (1,0,0.5), (0.5,0,1), (0,0,0.5), (0.5,0,0.02)):
            if verbose: print(f"\t {target_pos}", end="")
            env.set_target_position(np.array(target_pos))
            terminated, truncated, step_count = False, False, 0
            while step_count < max_steps_nb:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                if terminated: break
     
            dist_effect_target = norm(np.array(env.effector_pos) - target_pos)
            err.append(dist_effect_target)
            if verbose: print(f" {step_count} steps, dist: {dist_effect_target*100:.2f} cm ")    
    
        e_mean = np.array(err).mean()
        e_std  = np.array(err).std()
        error.append(err)
        print(f"\t e_mean, e_std: {e_mean*100:6.2f}, {e_std*100:6.2f} cm")
        if e_mean < err_mean:
            best_train = file
            err_mean = e_mean
            err_std  = e_std
    error = np.array(error)            
    return error, err_mean, err_std, list_files, best_train
    
    
def perf_nbSubStep_trajectoire(env, model_dir, file, DT, max_steps=500, verbose=0):
    '''
    For a gieven zip training file, we loop through many  numSubSteps values, and for each 
    we compute the mean distance between the robot end effector and the 5 target positions
    figuring a diamond. 
    '''
    q1_q2    = (113, -140)
    epsilon  = 1.e-3
    err_mean, err_std  = np.inf, np.inf
    error = []

    agent = PPO.load(model_dir /  'ZIP' / file)
    env._max_episode_steps = None
    
    for nSubSteps in range(1, 81, 1):            
        print(f">>> nSubStep: {nSubSteps:4d}", end="")
        err = []
        obs, _ = env.reset(options={"dt": DT, 
                                    "target_initial_pos": (0.5,0,0),
                                    "robot_initial_angle_deg": q1_q2,
                                    "randomize": False,
                                    "epsilon": epsilon,
                                    "numSubSteps": nSubSteps})    
        
        for target_pos in ((0.5,0.,0.02), (1,0,0.5), (0.5,0,1), (0,0,0.5), (0.5,0,0.02)):
            if verbose: print(f"\t {target_pos}", end="")
            env.set_target_position(np.array(target_pos))
            terminated, truncated, step_count = False, False, 0
            while step_count < max_steps:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                if terminated: break
          
            dist_effect_target = norm(np.array(env.effector_pos) - target_pos)
            err.append(dist_effect_target)
            if verbose: print(f"\t {step_count:3d} steps, dist: {dist_effect_target*100:.2f} cm ")    
    
        e_mean = np.array(err).mean()
        e_std  = np.array(err).std()
        error.append(err)
    
        print(f"\t e_mean:{e_mean*100:7.2f} cm, e_std:{e_std*100:7.2f} cm")
        
        if e_mean < err_mean:
            best_nSubSteps = nSubSteps
            err_mean = e_mean
            err_std  = e_std
                
    error = np.array(error)            
    return error, err_mean, err_std, best_nSubSteps
    
    
