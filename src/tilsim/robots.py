from threading import Lock
import time

import numpy as np
from numpy.random import default_rng

from tilsdk.localization import LocalizationService

# Helper methods

Rot = lambda t: np.array([[np.cos(t), -np.sin(t)],
                          [np.sin(t), np.cos(t)]], dtype=float)

##### Robot classes #####

class SimRobot:
    '''Simulated robot.'''

    def __init__(self, sim_config, timeout:float=0.5):
        '''
        Parameters
        ----------
        sim_config
            configuration file from the simulator.
        '''
        self.sim_config = sim_config
        self._pose = np.array(sim_config.start_pose, dtype=float)
        self._pose_lock = Lock()
        self._vel = np.array((0,0,0), dtype=float)
        self._last_changed = time.time()
        self._vel_lock = Lock()
        self.timeout = timeout

        self.rng = default_rng()

    def step(self, dt:float) -> None:
        '''Step the simulation.
        
        Parameters
        ----------
        dt : float
            Time since last simulation step.
        '''
        with self._pose_lock, self._vel_lock:
            vel = np.array([*(Rot(np.radians(self._pose[2]))@self._vel[:2]), self._vel[2]])
            self._pose += vel*dt

    @property
    def pose(self):
        with self._pose_lock:
            return self._pose

    @pose.setter
    def pose(self, value):
        with self._pose_lock:
            self._pose = np.array(value, dtype=float)
                
    @property
    def vel(self):
        with self._vel_lock:
            return self._vel

    @vel.setter
    def vel(self, value):
        with self._vel_lock:
            self._vel = np.array(value, dtype=float)
            self._last_changed = time.perf_counter()

    @property
    def last_changed(self) -> float:
        with self._vel_lock:
            return self._last_changed

    @property
    def noisy_pose(self):
        self._pose_lock.acquire()
        pose = self._pose
        self._pose_lock.release()

        # back out front and rear tag positions
        angle = np.radians(pose[2])
        half_dir_vec = self.sim_config.robot_phy_length/2*np.array((np.cos(angle), np.sin(angle)))
        front = pose[:2] + half_dir_vec
        back = pose[:2] - half_dir_vec
        
        # add noise
        # noisy_pos = pose[:2]
        noisy_pos = pose[:2] + self.rng.normal(0, self.sim_config.position_noise_stddev, size=2)
        #front += self.rng.normal(0, self.sim_config.position_noise_stddev, size=2)
        #back += self.rng.normal(0, self.sim_config.position_noise_stddev, size=2)
        
        #noisy_dir_vec = front-back
        #noisy_angle = np.degrees(np.arctan2(noisy_dir_vec[1], noisy_dir_vec[0]))
        original_angle = pose[2]
        #return  np.array([*((front+back)/2), original_angle])
        return  np.array([*(noisy_pos), original_angle])


class ActualRobot:
    '''Passthrough for actual robot.
    
    Uses pose information from a localization service
    instance and does not perform simulation.
    '''
    def __init__(self, loc_service: LocalizationService):
        '''
        Parameters
        ----------
        host : str
            Localization service host.
        port : int
            Localization service port.
        '''
        self.loc_service = loc_service
        self._pose = np.array((0,0,0))
        self._pose_lock = Lock()
        self.step(0) # initialize pose

    def step(self, dt:float) -> None:
        '''Step the simulation.
        
        For ActualRobot this gets latest pose from localization service.

        Parameters
        ----------
        dt : float
            Time since last simulation step.
        '''
        pose = self.loc_service.get_pose()
        if pose:
            with self._pose_lock:
                self._pose = np.array(pose)

    @property
    def pose(self):
        with self._pose_lock:
            return self._pose
