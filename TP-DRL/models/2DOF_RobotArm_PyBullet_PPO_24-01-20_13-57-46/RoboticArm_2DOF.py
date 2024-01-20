###########################################################################
# Copyright 2022 Jean-Luc CHARLES
# Created: 2022-07-29
# version: 1.2 - 3 Dec 2023 
# License: GNU GPL-3.0-or-later
###########################################################################

import time
import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import logger, spaces

from math import pi
import time, sys
import pybullet as p
import pybullet_data

from utils.tools import is_close_to,  welcome, plot_test
import rewards 

class RoboticArm_2DOF_PyBullet(gym.Env):
    '''
    To simulate a 2 DOF robotic arm using the Pybullet simulator.
    '''
    
    def __init__(self,
                 robot_urdf:str, 
                 target_urdf:str,
                 dt:float,
                 limit_q1:tuple  = (0, 180),
                 limit_q2:tuple  = (-180, 180),
                 init_robot_angles:tuple = (113, -140),
                 init_target_pos:tuple = (0.50, 0, 0.52),
                 t1_t2_conv = (800, 400),
                 reward:str = "reward",
                 clip_action:bool = False,
                 max_episode_steps:int = 500,
                 epsilon:float = 1e-3,
                 seed:int = None,
                 headless:bool = False,
                 verbose:int = 1):
        '''
        Parameters of the constructor of RoboticArm_2DOF:
            robot_urdf:      the URDF file that describes the robot architecture.
            target_urdf:     the URDF file that describes the target.
            dt:              timestep in seconds for the PyBullet simulation
            limit_q1:tuple   limits (min, max) of q1 in degrees
            limit_q2:tuple   limits (min, max) of q2 in degrees
            init_robot_angles: initial values of the angles  q1 & q2 of the robot  
            init_target_pos: the initial position (x,y,z) of the target to reach. Random 
                             perurbations can be added (see the 'reset' method).
            t1_t2_conv:      factors for converting actions in torques [Nm]      
            reward:          name of the reward function to use as a class method.
            clip_action:     whether to clip 'action' given by the neural network to keep its 
                             value in the 'action space' interval.
            max_episode_steps: int the max number of steps per episode (a.k.a the episode horizon).
            epsilon:         the threshold distance between robot end effector and target pos.
            seed:            integer value to set the seed of random generators
            headless:        wether to display 3D rendering or not.
                             For computing only headless=True (default value: True).
            verbose:         verbosity (displaying info), possible values:
                             0,1,2. Default value : 1.

            limits of q1 & q2 are given in degrees.
        '''

        # run the constructor of the base class:
        super().__init__()
        super().reset(seed=seed)
        
        self.robot_urdf        = robot_urdf
        self.target_urdf       = target_urdf
        self.seed_value        = seed    # seed for random generators

        self.dt                = dt
        self.limit_q1          = np.radians(limit_q1)
        self.limit_q2          = np.radians(limit_q2)
        self.t1_conv           = t1_t2_conv[0]
        self.t2_conv           = t1_t2_conv[1]
        self.robot_angles_deg  = init_robot_angles
        self.init_target_pos   = init_target_pos  # the target initial position (x,y,z) in [m].
        self.reward_func       = reward           # the name of the reward function
        self.clip_action       = clip_action      
        self.epsilon           = epsilon
        self.numSubSteps       = 50      # the number of sub-steps inside one pybullet simulation step.
        self.verbose           = verbose
        
        # id for the PyBullet simulator:
        self.robot_id          = None
        self.target_id         = None
        self.pc_id             = None     # the connexion id to PyBullet
        self.target_constraint = None     # id returned by pybullet.createConstraint
                  
        self.effector_pos      = None     # the robot end effector position (x,y,z) in [m].        
        self.observation_space = None     # Gym observation space
        self.action_space      = None     # Gym action space
        self.state             = None     # the robot state
        self.actual_target_pos = None     # np.ndarray, the actual target position: initial pos + possibly random
               
        self.cur_step          = 0        # the total number of steps done within an epiosde
                
        self._max_episode_steps = max_episode_steps
        if verbose >=1: print(f'[RoboticArm_2DOF_PyBullet.__init__] _max_episode_steps:{self._max_episode_steps}')
        
        # the indexes of the robot links  (see URDF file):                  
        self.base_link_index   = 0
        self.arm1_link_index   = 1
        self.arm2_link_index   = 2
        self.effect_link_index = 3
        
        # the indexes of the robot joints (see URDF file):
        self.motor_joints      = [1,2]                 

        # set the Gym action_space and observation_space:
        self.set_observation_space()
        
        self.min_action        = -1.0
        self.max_action        =  1.0
        self.set_action_space()

        # start the PyBullet simulator process:
        self.pybullet(headless)
       
        self.torque1 = []
        self.torque2 = []
        self.rewards = []
        self.target_pos = []   # to store all the target pos during the training
        self.ee_pos  = []      # to store all the end effector pos during the training
    
            
    def pybullet(self, headless):
        '''
        Run the pybullet simulation process.
        '''
        mode = p.DIRECT if headless else p.GUI 
        self.pc_id = p.connect(mode)              # or p.DIRECT for non-graphical version
        if not headless: 
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            welcome()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def set_action_space(self):
        '''
        Define the Gym action space: 
        - continuous in [-1,1] for motor1 driving q1, 
        - continuous in [-1,1] for motor2 driving q2.
        '''
        self.action_space = gym.spaces.Box(np.array([self.min_action, self.min_action]),
                                           np.array([self.max_action, self.max_action]))
        
    def set_observation_space(self):
        '''
        Define the observation space. The robot state vector is:
        (q1, q1_dot, q2, q2_dot, x_tg, z_tg, x_ee, z_ee) where
        - (x_tg, z_tg) is the target position
        - (x_ee, z_ee) is the robot end effector position.
        '''
        fmax = np.finfo(np.float32).max
        fmin = -fmax
        
        q_min = np.array([self.limit_q1[0], -10*np.pi, self.limit_q2[0], -10*np.pi, -2, 0, -2, 0])
        q_max = np.array([self.limit_q1[1],  10*np.pi, self.limit_q2[1],  10*np.pi,  2, 2,  2, 2])
        self.observation_space = gym.spaces.Box(q_min, q_max, dtype=np.float32)
    
    
    def updateState(self, step=False):
        '''
        Get the robot state and update the 'state' attribute [q1, q1_dot, q2, q2_dot].
        If <step> is True, request PyBullet simulation to make a step before.
        '''

        if step:
            p.stepSimulation(physicsClientId=self.pc_id)
            self.cur_step += 1
        
        # get the joints position and velocity:
        joints_state = p.getJointStates(self.robot_id, jointIndices=self.motor_joints, physicsClientId=self.pc_id)
        joints_pos   = joints_state[0][0], joints_state[1][0]
        joints_veloc = joints_state[0][1], joints_state[1][1]
    
        self.state = [joints_pos[0], joints_veloc[0], joints_pos[1], joints_veloc[1]]
    
        if self.verbose >= 3:
            print(f"\njoints_pos: {joints_pos}")
            print(f"joints_veloc.: {joints_veloc}")
                    
        # add the target position to the state vector:
        target_pos_list = self.actual_target_pos.tolist()
        self.state += [target_pos_list[0], target_pos_list[2]] 
        
        # get the robot and effector position:
        effector_state = p.getLinkState(self.robot_id, self.effect_link_index, physicsClientId=self.pc_id)
        self.effector_pos = effector_state[0]
        
        self.state += [self.effector_pos[0], self.effector_pos[2]]
        
        return np.array(self.state).astype('float32')
    
    def compute_done_reward(self, action):
        '''
        Compute the value of the reward and the 'done' boolean.
        '''
        q1, q1_dot, q2, q2_dot, x_tg, z_tg, x_ee, z_ee = self.state
        
        terminated, truncated = False, False
        
        # If the effector position is close to the target position, the episode is done:    
        if is_close_to(self.actual_target_pos, self.effector_pos, self.epsilon):
            terminated = True
            reward     = 10  #JLC: was 10!
            if self.verbose >=1 : print(f" step: {self.cur_step} done by close_to")
        # limit the total number of steps:
        elif self._max_episode_steps is not None and self.cur_step >= self._max_episode_steps:
            truncated  = True
            reward     = 0
            if self.verbose >=1 : print(f" step: {self.cur_step} done by max_step")
        
        if terminated == False and truncated == False:
            reward_instr = f"rewards.{self.reward_func}(self, action)"
            reward = eval(reward_instr) # eval() generates Python code

        #print(f"\r{reward:.2f}", end="")
        return terminated, truncated, reward
    
    def step(self, action):
        '''
        Apply actions to the environment (torques) and run one step of simulation.
        '''
            
        try :
            assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))
        except :
            if self.clip_action: np.clip(action, self.min_action, self.max_action)

        # torque1 & torque2 are the torques applied to motor 1 & 2
        # between -1 Nm and 1 Nm:
        torque1, torque2 = action
        self.torque1.append(torque1)
        self.torque2.append(torque2)
            
        torque1 *= self.t1_conv
        torque2 *= self.t2_conv
        
        #
        # Now, computes terminated, truncated and the reward:
        #
        terminated, truncated = False, False

        if self.robot_id is None:
            reward = 0
        else:
            # step
            p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                        jointIndices=self.motor_joints,
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0., 0.],
                                        physicsClientId=self.pc_id)
            
            p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                        jointIndices=self.motor_joints,
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=[torque1, torque2],                                        
                                        physicsClientId=self.pc_id)

            p.stepSimulation(physicsClientId=self.pc_id)
            self.cur_step += 1        

            self.updateState() # update the attribute self.state
            terminated, truncated, reward = self.compute_done_reward(action)
        
        self.rewards.append(reward)
        #print("state, action", self.state, action)
        return np.array(self.state).astype('float32'), reward, terminated, truncated, {}

    def reset(self, seed:int = None, options:dict =None):
        '''
        Resets simulation and spawn every object in its initial position.
        
        Parameters:
        - seed: if not None, its value is used to reset the RNG.
        - options: optionnal dictionary to give:
            -'dt': to change the self.dt time step of the PyBullet simulation
            -'robot_reset_angle_deg': the initial values for the robot q1 & q2 angles
            -'target_initial_pos': the initial target position
            -'randomize': whether to add or not random perturbation to the robot angles
                         and the traget position.     
            -'epsilon': to change the self.epsilon value.
            -'numSubSteps': to change the numSubSteps value.
            
        Return: the observation (the environement state) and possibly a dictionary
                of debug/info data.
        '''

        if self.verbose >= 1: print(f"[RoboticArm_2DOF_PyBullet.reset] last #steps : {self.cur_step}")
    
        # reset the random generator (rng) with the fixed seed if needed:
        if seed != None: 
            self.seed_value = seed
            super().reset(seed=seed)
    
        dt          = None
        epsilon     = None
        numSubSteps = None        
        
        # retrieve the initial robot angles and target position given at constructor
        robot_initial_angle_deg = self.robot_angles_deg
        target_initial_pos      = self.init_target_pos   
        randomize               = True
        
        if options:
            dt = options.get("dt", None) 
            robot_initial_angle_deg = options.get("robot_initial_angle_deg", robot_initial_angle_deg)
            target_initial_pos = options.get("target_initial_pos", target_initial_pos)
            randomize = options.get("randomize", randomize)
            epsilon = options.get("epsilon", None) 
            numSubSteps = options.get("numSubSteps", None)
                                               
        self.cur_step = 0
        self.prev_dist_effect_target = None
        
        if dt is not None : self.dt = dt
        if epsilon is not None: self.epsilon = epsilon
        #if numSubSteps is not None: self.numSubSteps = numSubSteps
                
        # After a PyBullet reset, every parameter has to be set again:
        p.resetSimulation(physicsClientId=self.pc_id)
        if numSubSteps != None: p.setPhysicsEngineParameter(numSubSteps=numSubSteps, physicsClientId=self.pc_id)
        p.setTimeStep(self.dt, physicsClientId=self.pc_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.pc_id)
        p.resetDebugVisualizerCamera(cameraDistance=2., cameraYaw=0, cameraPitch=0., 
                                     cameraTargetPosition=[0, 0, 1.], physicsClientId=self.pc_id)
        
        # Load the ground plane:
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.pc_id)
        
        # set the target position with the initial position plus a random perturbation in 
        # the two directions x and z:
        self.actual_target_pos = np.array(target_initial_pos)
        if randomize: 
            self.actual_target_pos += self.np_random.uniform(-0.51, 0.51, (3,))
        self.actual_target_pos[1] = 0.
        # z position should be gretaer then 1 cm:
        if self.actual_target_pos[2] < 0.01 : self.actual_target_pos[2] = 0.01
        # load the URDF of the target (defines the 'target_id' attribute):
        self.load_target(self.actual_target_pos)
        self.target_pos.append(self.actual_target_pos)
                
        # Load the robot URDF:
        start_pos = [0, 0, 0]
        start_orn = p.getQuaternionFromEuler((0, 0, 0), physicsClientId=self.pc_id)
        self.robot_id = p.loadURDF(self.robot_urdf, start_pos, start_orn, physicsClientId=self.pc_id)
                
        # set the robot angles with the initial values + some small random perturbation:
        angles_rad = np.radians(np.array(robot_initial_angle_deg))
        if randomize: 
            angles_rad += self.np_random.uniform(-0.6, 0.6, (2,))

        # Move the robot to the start position:
        p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                    jointIndices=self.motor_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles_rad,
                                    physicsClientId=self.pc_id)
                                    
        # loop to reach the position thanks to the physical engine computation.
        # Quit the loop when the robot is close to the desired position defined by 'angles_rad'
        sub_step = 0
        while(True):
            p.stepSimulation(physicsClientId=self.pc_id)
            joints_state = p.getJointStates(self.robot_id, 
                                            jointIndices=self.motor_joints, 
                                            physicsClientId=self.pc_id)
            joints_pos = joints_state[0][0], joints_state[1][0]
            sub_step += 1
            if is_close_to(angles_rad, joints_pos, self.epsilon): break
        
        # update the state of the robot:    
        obs = self.updateState() 
        # set q1_dot and q2_dot to zero:
        obs[1] = 0.
        obs[3] = 0.
        self.state = obs.tolist()
        self.ee_pos.append(obs[6:])
        
        if self.verbose >= 2:
            print(f"[RoboticArm_2DOF_PyBullet.reset] "
                  f"q1:({obs[0]*180/pi:.1f}°, {obs[1]*180/pi:.1f}°/s), "
                  f"q2:({obs[2]*180/pi:.1f}°, {obs[3]*180/pi:.1f}°/s), "
                  f"Target:({obs[4]:.3f},{obs[5]:.3f}) m, "
                  f"EndEff:({obs[6]:.3f},{obs[7]:.3f}) m, "
                  f"after {sub_step} substeps")                 
        
        return obs, {}
    
    def load_target(self, pos: np.ndarray):
        '''
        Load the target using its URDF file.
        '''
        
        # load the target URDF:
        start_pos = [0, 0, 0.0]
        start_orn = p.getQuaternionFromEuler((0, 0, 0), physicsClientId=self.pc_id)
        self.target_id = p.loadURDF(self.target_urdf, start_pos, start_orn, physicsClientId=self.pc_id)
        
        # Move the target to the desired position:
        self.target_constraint = p.createConstraint(self.target_id, -1, -1, -1, p.JOINT_FIXED, 
                                                    jointAxis=[0, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=pos.tolist(),
                                                    physicsClientId=self.pc_id)

    def set_target_position(self, new_pos:np.ndarray):
        '''
        Move the target to a given position
        Parameters:
        - new_pos: the ndarray of the 3D position coordinates.
        '''
        
        # Move the target to the desired position:
        
        p.removeConstraint(self.target_constraint, physicsClientId=self.pc_id)
        
        self.target_constraint = p.createConstraint(self.target_id, -1, -1, -1, p.JOINT_FIXED, 
                                                    jointAxis=[0, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=new_pos.tolist(),
                                                    physicsClientId=self.pc_id)
        self.actual_target_pos = new_pos
        
        
    def testAngleControl(self, dt:float, angle_amplitude:float = pi/5):
        '''
        Run a test by moving the robot motors by a given angle amplitude
        Parameters:
        - dt: the time step of the simulation [s]
        - angle_amplitude: the amplitude with which to move the two motors angle.
        '''

        self.dt = dt
        
        States = []
        state  = self.reset()

        p.setPhysicsEngineParameter(numSubSteps=self.numSubSteps, physicsClientId=self.pc_id)
        
        print(f"[testAngleControl] Ready to run the test with timestep={self.dt:8.2e} s")
        #input(f"[testAngleControl] Press ENTER to launch the free run simulation...")
        
        list_angle = angle_amplitude*np.sin(np.linspace(0, 1., 60)*2*pi)

        simul_time = 0
        
        for a in list_angle:    
            if self.verbose >= 2: print(f"\r{a*180/pi:.2f}°", end="")
            angles = [113*pi/180+a, -140*pi/180-a]
            p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                        jointIndices=self.motor_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=angles,
                                        physicsClientId=self.pc_id)
            sub_step = 0
            while(True):
                # make some iterations to get the robot's EndEffector close to the target position:
                p.stepSimulation(physicsClientId=self.pc_id)
                joints_state = p.getJointStates(self.robot_id, jointIndices=self.motor_joints, physicsClientId=self.pc_id)
                joints_pos = joints_state[0][0], joints_state[1][0]
                sub_step += 1
                if sub_step > 1: print("\rsub_steps:", sub_step)
                if is_close_to(angles, joints_pos, self.epsilon): break
            
            self.cur_step += 1        
            simul_time = self.cur_step*self.dt
            # observe (sets the attribute self.state):
            state = self.updateState().tolist()
            #print(state)
            data = [simul_time] + [sub_step] + state + [a] + angles
            States.append(data)
            time.sleep(self.dt)
            
        return np.array(States)

    def testq1q2(self, dt:float, q1q2_deg:list = [90, -90]):
        '''
        Move the robot to a position defined by the two angles q1 and q2.
        Parameters:
        - dt : the time step of the simulation
        - q1q2_deg: the liste of the two angles [q1, q2] (in degrees).
        '''

        self.dt = dt
        
        States = []
        state, _  = self.reset()

        p.setPhysicsEngineParameter(numSubSteps=self.numSubSteps, physicsClientId=self.pc_id)
        
        print(f"[testq1q2] Ready to run the test with timestep={self.dt*1e3:.1} ms"
              f"This is the Robot position after reset")
        input(f"[testq1q2] Press ENTER to move the robot to q1={q1q2_deg[0]:.1f}° and q2={q1q2_deg[1]:.1f}°...")
        
        simul_time = 0
        angles = [q1q2_deg[0]*pi/180, q1q2_deg[1]*pi/180]
        
        p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                    jointIndices=self.motor_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=angles,
                                    physicsClientId=self.pc_id)
        sub_step = 0
        
        while(True):
            p.stepSimulation(physicsClientId=self.pc_id)
            joints_state = p.getJointStates(self.robot_id, jointIndices=self.motor_joints, physicsClientId=self.pc_id)
            joints_pos = joints_state[0][0], joints_state[1][0]
            sub_step += 1
            print("\rsub_steps:", sub_step, end="")
            if is_close_to(angles, joints_pos, self.epsilon): break
        
        self.cur_step += 1        
        
        # observe (sets the attribute self.state):
        state = self.updateState()
        with np.printoptions(formatter={'float':'{:.3f}'.format}):    
            print(f"\nrobot's state: {state}")
            
    def close(self):
        '''
        To close properly the PyBullet simulator session and all what is connected to the Environment
        '''
        
        try:
            p.disconnect(physicsClientId=self.pc_id)
        except:
            print("not connected...")


if __name__ == '__main__':
    
    # scene path relative to the 'project root' directory:
    ROBOT  = "./urdf//RoboticArm_2DOF_2.urdf"
    TARGET = "./urdf/target.urdf"
    env    = RoboticArm_2DOF_PyBullet(robot_urdf  = ROBOT,
                                      target_urdf = TARGET,
                                      dt=1./240,
                                      headless = None,
                                      verbose=0)
    
    #data = env.testAngleControl(0.05)
    #data = data.astype(float)
    
    env.testq1q2(10e-3)    
    input("press ENTER to end....")
    
    # always end with env.close() to close the simulation and all what is connected to: 
    env.close()
    
