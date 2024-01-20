###########################################################################
# Copyright 2022 Jean-Luc CHARLES
# Created: 2022-07-29
# version: 1.2 - 3 Dec 2023
# License: GNU GPL-3.0-or-later
###########################################################################

import pybullet as p
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import pi
import time
import os
from stat import ST_MTIME

def get_files_by_date(directory):
    '''
    find files in a directory and sort the files by date.
    '''
    files = []
    for f in os.listdir(directory):
        file = os.path.join(directory, f)
        if os.path.isfile(file): files.append((os.stat(file)[ST_MTIME], f))
    files.sort()
    return  [f for s,f in files]

def moving_average(x, n):
    '''
    Computes the moving average of data in table x using a window of n data points.
    '''
    ma = np.copy(x)
    ma[n-1:] = np.convolve(x, np.ones(n), 'valid') / n
    return ma

def welcome():
    '''
    Display some useful PyBullet shortcuts.
    '''
    print("\n"+"#"*80)
    print("# Welcome to this practical session with Pybullet & URDF.")
    print("# Pybullet windows shortcuts:")
    print("#    G: close/open the tabs")
    print("#    W: switch between solid/wireframe rendering")
    print("#    J: show/hide links & joints frames as RGB lines (with wireframe rendering activated)")
    print("#    K: show/hide joint axes as a black line         (with wireframe rendering activated)")
    print("#    A: show/hide collision boxes                    (with wireframe rendering activated)")
    print("#    CTRL+left_clic  : rotate the robot")
    print("#    CRTL+midlle_clic: translate the robot")
    print("#    Mouse_wheel: zoom/unzoom")
    print("#"*80+"\n")
    
def is_close_to(x, y, epsilon: float=1.e-3, verbose=0):
    '''
    Returns True if ||x-y|| <= epsilon
    '''
    n = norm(np.array(x) - np.array(y))
    if verbose: 
        with np.printoptions(precision=3, suppress=True):
            print(f"||{np.array(x)}-{np.array(y)}|| gives {n:.4f}, to compare to {epsilon}")
    return  n <= epsilon

def move_to(botId:int, 
            joints:tuple, 
            target:tuple, 
            verbose:int=0, 
            wait:str="Press ENTER for next position"):
    '''
    Use the PyBullet simulator to move the robot arms: the robot target position is given  
    as two angles (q1,q2) [rad]
       
    Parameters:
      botId:int:    the id of the robot
      joints:tuple: the list of the joint indexes to control
      target:tuple: the list of the target angles to reach
      verbose:int:  optional, used to tune the function verbosity (0: no display)
      wait:str:     optional; if not empty, the function will wait for a user input after moving the joints.
       
    Return: None.
    '''
    # Display only 3 digits after decimal separator for numpy objects:
    np.set_printoptions(precision=3)

    angles = np.degrees(target)
    print(f"\nTarget robot position: q1={angles[0]:.0f}° and q2={angles[1]:.0f}° ")

    # give the target position control:
    p.setJointMotorControlArray(bodyIndex=botId,
                                jointIndices=joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=target)
    
    # The following loop shows how the joints reach the target position under the physical engine computation.
    # The loop breaks when the robot is close to the target position:
    step = 1
    while(True):
        p.stepSimulation()
        joints_state = p.getJointStates(botId, jointIndices=joints)
        # getJointStates return a list of joint state.
        # Each joint state is a liste: [ 3D_vector_of_posiion, 3D_vector_of_velocities, ...]
        #   joints_state[0][0] is the position of the 1rst joint in the list (the angle for a revolute),
        #   joints_state[1][0] is the position of the 2nd joint in the list (the angle for a revolute).
        joints_pos = joints_state[0][0], joints_state[1][0]
        if verbose >=2 :
            print(f"joints_pos: ({joints_pos[0]:.3f}, {joints_pos[1]:.3f})")
        if is_close_to(target, joints_pos, verbose=verbose) : break
        step += 1
    if verbose >=1 : print(f"Target position reached in {step} simulation step")
    if wait: input(wait)
    
def display_joint_properties(botId:int, jointIndex:int=None):
    '''
    Display the properties of all joints or a specific joint if jointIndex is not None.
    '''
    
    labels = ("jointIndex", "jointName", "jointType", "qIndex", "uIndex", "flags", 
              "jointDamping", "jointFriction", "jointLowerLimit", 
              "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName", "jointAxis")
    labSelect = ("jointName", "qIndex", "uIndex", "jointDamping", "jointFriction", "jointLowerLimit", 
            "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName", "jointAxis")

    for i in range(p.getNumJoints(botId)):
        if jointIndex is not None and i != jointIndex:
            continue
        else:
            infos = p.getJointInfo(botId, i)
            infoDict = { lab:prop for lab, prop in zip(labels, infos)}
            print(f"Infos on joint index <{i}>")
            for key in labSelect:
                value = infoDict[key]
                print(f"\t{key:16s}:{value}")
            print()

def display_link_properties(botId:int):
    '''
    Display the properties of all links.
    '''
    # Display only 3 digits after decimal separator for numpy objects:
    np.set_printoptions(precision=3)
    
    labels=("linkWorldPosition", "linkWorldOrientation", "localInertialFramePosition", 
            "localInertialFrameOrientation", "worldLinkFramePosition", "worldLinkFrameOrientation", 
            "worldLinkLinearVelocity", "worldLinkAngularVelocity")
    labSelect = ("linkWorldPosition", "linkWorldOrientation")

    for i in range(4):
        print(f"Infos on link index <{i}>")
        state = infos = p.getLinkState(botId, i)
        infoDict = {lab:prop for lab, prop in zip(labels, state)}
        for key in labSelect:
            values = infoDict[key]
            print(f"\t{key:21s}: ("+', '.join([f"{v:6.3f}" for v in values])+')')
        print()


def test_training(agent, 
                  env, 
                  DT:float, 
                  pts, 
                  max_steps_nb:int = 50, 
                  epsilon=1e-3, 
                  nSubSteps=50):
    '''
    agent: the trained network
    env: the robot
    DT:  the time step for the Pybullet simulation
    pts: the array of points that make the trajectory
    max_steps_nb:int: the max number of steps to reach the target
    epsilon: the distance threshold between teh end effector end teh target
    nSubSteps: the number of substep for one Pybullet step
    '''

    q1_q2 = (113, -140)
    
    obs, _ = env.reset(options={"dt": DT, 
                                "target_initial_pos": (0.5,0,0),
                                "robot_initial_angle_deg": q1_q2,
                                "randomize": False,
                                "epsilon": epsilon,
                                "numSubSteps": nSubSteps})

    # First, position the end effector on the firts point:
    target_pos = (1, 0, 0.5)   # JLC = pts[0]
    env.set_target_position(np.array(target_pos))

    terminated, step_count, rewards, actions = False, 0, [], []
    while step_count < 5*max_steps_nb:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        step_count += 1
        if terminated: break
    #time.sleep(0.1)

    # Now, explore the trajectory by segments:
    error = []
    for target_pos in pts:
        env.set_target_position(target_pos)
        terminated, step_count = False, 0
        while step_count < max_steps_nb:
            if not terminated:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, _, info = env.step(action)
                dist_effect_target = norm(np.array(env.effector_pos) - target_pos)
                error.append(dist_effect_target)
            #time.sleep(env.dt)
            step_count += 1

    error = np.array(error)
    print(f"\ne_av: {100*error.mean():.2f} cm,"
          f" e_std:{100*error.std():.2f} cm, abs(e)_max: {100*max(abs(error)):.2f} cm")
    
    return error

def sample_line(p1p2: tuple, nb_sampling_pts: int):
    '''
    To sample a line defined by 2 points into a sequence of small consecutive segments.
    parameters:
      p1p2: the tuple of the 2 points ((x1,y1), (x2,y2)). Each point is a tuple of coordinate (x,z).
      nb_sampling_pts: the number of sampling points.
      
    return:
      points: the ndarray of floats of shape (n_sample, 3). Each row is the 3 coords (x,y,z) of the
      intermediate points.
      dl: th average length (norm) of the segments.
    '''
    x1, z1 = p1p2[0]
    x2, z2 = p1p2[1]
    
    dl = norm(np.array(p1p2[0]) - np.array(p1p2[1]))/(nb_sampling_pts - 1)
    
    a = (z2 - z1) / (x2 - x1)
    b = z1 - a*x1
    X = np.linspace(x1, x2, nb_sampling_pts)
    Z = a*X + b
    points = np.ndarray((nb_sampling_pts, 3))
    points[:,0] = X
    points[:,1] = 0
    points[:,2] = Z
    return points, dl
    
def sample_traj4pts(trajectory, nb_pts_per_line):
    '''
    To sample the points of the 4 lines that define a trajectory.
    parameters:
      trajectory: the tuple of the 4 pair of points that define the lines of the closed trajectory.
      nb_sampling_pts_per_line: the number of sampling points per line of the trajectory.
      
    return:
      pts: the ndarray of the coordinates (x,y,z) of the sampled points
      dl: the average of the segments length between two conscutive points.
    '''
    list_pts = []   
    list_dl  = []
    for p1p2 in trajectory:
        pts, dl = sample_line(p1p2, nb_pts_per_line)
        pts = pts[:-1].tolist()
        list_pts += pts
        list_dl.append(dl)
    pts = np.array(list_pts)      
    dl  = np.array(list_dl).mean()
    
    return pts, dl

def plot_test(data):
        
    # plt.figure(figsize=(5,10))
    # data = [simul_time] + [steps] + state + [a] + angles
    # the robot state vector is :
    # (q1, q1_dot, q2, q2_dot, x_tg, z_tg, x_ee, z_ee) where
    # - (x_tg, z_tg) is the target position
    # - (x_ee, z_ee) is the robot end effector position.

    fig = plt.figure(figsize=(10,8)) 
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.07, hspace=0.7)
    fig.suptitle(r"RobotArm_2DOF: test driving $q_1$ and $q_2$", size=14)
    axe1 = plt.subplot2grid((4,1), (0,0))
    axe2 = plt.subplot2grid((4,1), (1,0))
    axe3 = plt.subplot2grid((4,1), (2,0), rowspan=2)

    time   = data[:,0]
    steps  = data[:,1]
    state  = data[:,2:10]
    effect = state[:,6:]
    a      = np.degrees(data[:,10])
    angles = np.degrees(data[:,11:])

    axe1.set_title(r"Robot angles $q_1$ and $q_2$")
    l1, = axe1.plot(time, np.degrees(state[:,0]), "b.")
    l2, = axe1.plot(time, angles[:,0], ":k")
    axe1.set_xlabel("Time [s]")
    axe1.set_ylabel("angle [°]", color="b")
    axe1.grid()
    axe1.set_ylim(50,180)
    axe11= axe1.twinx()
    l3, = axe11.plot(time, np.degrees(state[0:,2]), "r.")
    l4, = axe11.plot(time, angles[:,1], ":k")
    axe11.set_ylim(-190, -60)
    axe11.set_ylabel("angle [°]", color= "r")
    axe1.legend([l1, l3, l2], [r'$q_1$', r'$q_2$', 'drive'], loc="upper right", ncol=3)
    #axe11.legend([l2, l4], ['drive1', 'drive2'], loc="lower right")

    axe2.set_title(r"Robot angle velocities $\dot{q}_1$ and $\dot{q}_2$")
    l1, = axe2.plot(time, np.degrees(state[:,1]), "b.", )
    l2, = axe2.plot(time, np.degrees(state[0:,3]), "r.")
    axe2.set_xlabel("Time [s]")
    axe2.set_ylabel("angle veloc.[°/s]")
    axe2.legend([l1, l2], [r"$\dot{q}_1$", r"$\dot{q}_2$"], loc="center right", ncol=1)
    axe2.grid()

    axe3.set_title(r"Robot end effector trajectory")
    axe3.plot(effect[:,0], effect[:,1],"m.")
    axe3.set(xlim=(0,1.3), ylim=(0,.8))
    axe3.set_aspect('equal')
    axe3.grid()
    axe3.set(xlabel=("X position [m]"), ylabel=("Z position [m]"))

    plt.savefig("RobotArm_2DOF_test.png")
    plt.show()
        
