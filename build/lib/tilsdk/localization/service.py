import urllib3
from .types import *
import json
import base64
import matplotlib.pyplot as plt
import io
from typing import List, Tuple
import logging

class LocalizationService:
    '''Communicates with localization server to obtain the arena's static map and the robot's estimated pose.
    '''

    def __init__(self, host:str='localhost', port:int=5566):
        '''
        Parameters
        ----------
        host : str
            Hostname or IP address of localization server.
        port: int
            Port number of localization server.
        '''
        self.url = 'http://{}:{}'.format(host, port)
        self.manager = urllib3.PoolManager()
        logging.getLogger('Localization').info(f"Localization Service connecting to {self.url}.")

    def get_map(self) -> SignedDistanceGrid:
        '''Get a grid-based representation of the of the map.
        
        Grid elements are square and represented by a float. Value indicates distance from nearest
        obstacle. Value <= 0 indicates occupied, > 0 indicates passable.
        Grid is centered-aligned, i.e. real-world postion maps to center of grid square.
        
        Returns
        -------
        grid : SignedDistanceGrid
            Signed distance grid.
        '''

        response = self.manager.request(method='GET',
                                        url=self.url+'/map')

        data = json.loads(response.data)

        grid = base64.decodebytes(data['map']['grid'].encode('utf-8'))

        img = plt.imread(io.BytesIO(grid))
        grid = SignedDistanceGrid.from_image(img, data['map']['scale'])

        return grid

    def get_pose(self) -> RealPose:
        '''Get real-world pose of robot.
        
        Returns
        -------
        pose : RealPose
            Pose of robot.
        '''

        response = self.manager.request(method='GET',
                                        url=self.url+'/pose')

        if response.status != 200:
            logging.getLogger('Localization Service').debug('Could not get pose.')
            return None, None

        data = json.loads(response.data)

        pose = RealPose(
            x=data['pose']['x'] if data['pose']['x'] > 0 else 0,
            y=data['pose']['y'] if data['pose']['y'] > 0 else 0,
            z=data['pose']['z']
        )
        
        return pose