import urllib3
import json
import base64
import logging 
from typing import List, Any, Tuple, Union
from pathlib import Path

import cv2

from tilsdk.localization.types import RealPose
from .response_utils import save_zip

class ReportingService:
    '''Communicates with reporting server to submit reports.'''

    def __init__(self, host:str='localhost', port:int=5000):
        '''
        Parameters
        ----------
        host
            Hostname or IP address of reporting server.
        port
            Port number of reporting server.
        '''

        self.url = 'http://{}:{}'.format(host, port)
        self.manager = urllib3.PoolManager()
        logging.getLogger('Reporting').info(f"Reporting Service connecting to {self.url}.")


    def report_situation(self, img:Any, pose: RealPose, answer, save_dir: Union[str, Path]):
        '''Report answer for the Friend or Foe (Visual) task, namely the situation about 
        whether 'hostage' or 'suspect' is in the scene. This method will attempt to receive
        a zipped folder from the Reporting Server and unzip and save the folder into the
        ``save_dir``.

        Parameters
        ----------
        img : ndarray
            cv2 image taken by robot and drawn with bounding boxes of detected objects.
        pose : RealPose
            Robot's pose at the time when this picture was taken.
        answer : str
            Who is in the scene: "hostage", "suspect" or "none".
        save_dir: str or Path
            The directory for saving the returned folder of audio files into. 
            The Reporting server returns some audio files for the next task, 
            "Friend or Foe (Audio)".

        Returns
        -------
        save_path : str
            The path to the folder of contents retrieved from the server.
        '''
        validate_reid_submission(answer)
        
        _, encoded_img = cv2.imencode('.jpg',img)
        base64_img = base64.b64encode(encoded_img).decode("utf-8")

        response = self.manager.request(method='POST',
                                        url=self.url+'/report_situation',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'image': base64_img,
                                            'pose': pose,
                                            'situation': answer
                                        }),
                                        preload_content = False)
        if response.status == 200:
            save_path = save_zip(save_dir, response)
        else:
            raise Exception(f"Bad Response from server. Response code: {response.status}. " + 
                            f"Check inputs are correct or that the server is up.")

        return save_path

    def report_audio(self, pose: RealPose, answer: str, save_dir: Union[str, Path]):
        ''' Report answer for the Friend or Foe (Audio) task.

        Parameters
        ----------
        pose : RealPose
            Robot's pose at the time when this corresponding checkpoint's task was being done.
        answer: str
            Answer for the Friend or Foe (Audio) task.
            The expected format is <Filename of Audio File w/o file extension>_<Team Name>_<Member ID>
            e.g. "audio1_My Team Name is Great_MemberA".
        save_dir : str or Path
            The directory for saving a returned folder of audio files into. The Reporting server
            returns some audio files for the next task, "Decoding Digits".

        Returns
        -------
        save_path : str
            The path to the folder of contents retrieved from the server.
        '''
        validate_speakerid_submission(answer)
        
        response = self.manager.request(method='POST',
                                        url=self.url+'/report_audio',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'pose':pose,
                                            'chosen_audio': answer
                                        }),
                                        preload_content = False)
        if response.status == 200:
            save_path = save_zip(save_dir, response)
        else:
            raise Exception(f"Bad Response from server. Response code: {response.status}. " + 
                            f"Check inputs are correct or that the server is up.")

        return save_path


    def report_digit(self, pose:RealPose, answer: Tuple):
        ''' Report answer for the Decoding Digits task.
        
        Parameters
        ----------
        pose : RealPose
            Robot pose where targets were seen.
        answer : Tuple
            A tuple of digits representing the digits decoded from a series of audio files.

        Returns
        -------
        pose : RealPose
            Target pose of the next checkpoint.
        '''
        response = self.manager.request(method='POST',
                                        url=self.url+'/report_digit',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'pose': pose,
                                            'recover_digits': answer
                                        }),
                                        preload_content = False)
        if response.status == 200:
            pose = eval(response.data)
        else:
            raise Exception(f"Bad Response from server. Response code: {response.status}. " + 
                            f"Check inputs are correct or that the server is up.")

        return pose


    def start_run(self):
        ''' Inform scoring server that the robot is starting the run.
        This **must** be called before making submissions to the scoring server and before the
        robot starts moving.
        
        Parameters
        ----------
        
        Returns
        -------
        response : Flask.response
            http response.
        '''
        response = self.manager.request(method='GET',
                                        url=self.url+'/start_run')
        return response

    def end_run(self):
        '''Tells the scoring server that the robot is terminating its run.
        Call this **only** after receiving confirmation from the scoring server that you
        have reached the maze goal.
        '''
        response = self.manager.request(method='GET',
                                        url=self.url+'/end_run')
        return response
    
    def check_pose(self, pose:tuple):
        ''' Checks the status of the ``pose``.
        
        Returns
        -------
        str or RealPose
            "Goal Reached" if pose is considered near enough to maze's end goal.
            "Task Checkpoint Reached" if pose is near enough to Task Checkpoint.
            "Not An Expected Checkpoint" if ``pose`` is not a goal, task or detour checkpoint.
            A RealPose of (x, y, heading in degrees) representing the next checkpoint if ``pose`` is a detour checkpoint.
        '''
        response = self.manager.request(method='GET',
                                        url=self.url+'/check_pose',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'pose': pose
                                        }),
                                        preload_content = False)
        
        if response.status == 200:
            data = response.data.decode('utf-8')
            if data == "End Goal Reached":
                return data
            elif data == "Task Checkpoint Reached":
                return data
            else:  # interpret this string as a 3-tuple of pose information: x, y, heading.
                pose_tup = eval(data)
                return pose_tup
        elif response.status == 300:
            data = response.data.decode('utf-8')
            if data == 'Not An Expected Checkpoint':
                return data
            elif data == "You Still Have Checkpoints":
                return data
            else:
                raise Exception(f"Unhandled Response. Response code: {response.status}.")
        else:
            raise Exception(f"Bad Response from server. Response code: {response.status}. " + 
                            f"Try checking that inputs are correct or that the server is up.")


def validate_speakerid_submission(answer: str):
    '''
    Validate string submission for the Friend or Foe (Audio) task.
    The expected format is <Filename of Audio File w/o file extension>_<Team Name>_<Member ID>
    e.g. "audio1_My Team Name is Great_MemberA".
    '''
    answer = answer.lower()
    parts = answer.split("_")
    assert len(parts) == 3
    assert '.' not in parts[0]  # should not look like it contains a file extension.
    
    
def validate_reid_submission(answer: str):
    '''
    Validate string submission for ReID Friend or Foe (Visual) task.
        
    The expected answer is "hostage", "suspect" or "none".
    '''
    assert type(answer) == str
    assert answer in ["hostage", "suspect", "none"]
