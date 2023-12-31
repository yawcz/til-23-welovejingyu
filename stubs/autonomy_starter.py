# SAMPLE AUTONOMY CODE FOR TIL2023. This is not the most optimal solution.
# However, it is a fully-running codebase with major integrations done for you
# already so that you can concentrate more on improving your algorithms and models 
# and less on integration work. Participants are free to modify any thing in the 
# "stubs" folder. You do not need to modify the "src" folder. Have fun!

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import List
import yaml

from librosa import load as load_audio

# Import necessary and useful things from til2023 SDK
from tilsdk import *                                                # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage     # import optional useful things
from tilsdk.reporting import save_zip                               # to handle embedded zip file in flask response

# Import your code
from planner import Planner
from planner import InvalidStartException, NoPathFoundException    # Exceptions for path planning.

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

##### HELPER FUNCTIONS #####

def load_audio_from_dir(save_path: str):
    audio_dict = {}
    for item in os.scandir(save_path):
        if item.is_file() and item.name.endswith(".wav"):
            audio_fpath = Path(os.fsdecode(item.path)).as_posix()
            logging.getLogger('Main').info(f"loading {audio_fpath}...")
            #with open(audio_fpath) as audio_file:
            audio_dict[item.name] = load_audio(audio_fpath, sr=16000)  # returns audio waveform, sampling rate.
    return audio_dict  # e.g. {'audio1.wav': <return type of librosa.load>}


def reid(reid_service, hostage_img, suspect_img, img):
    # pass image to re-id service, check whether Suspect or Hostage was found.
    logging.getLogger('reid').info(f"hostage img shape {hostage_img.shape}.")
    logging.getLogger('reid').info(f"suspect img shape {suspect_img.shape}.")
    logging.getLogger('reid').info(f"robot's photo shape {img.shape}.")
    
    sus_bbox = reid_service.targets_from_image(img, suspect_img)
    hostage_bbox = reid_service.targets_from_image(img, hostage_img)
    logging.getLogger('Main').info(f"suspect?: {sus_bbox};\n hostage?: {hostage_bbox}.")
    return sus_bbox, hostage_bbox


def take_photo(robot, photo_dir):
    '''Get robot to take photo and save it into photo_dir'''
    robot.camera.start_video_stream(display=False, resolution='720p')
    img = robot.camera.read_cv2_image(strategy='newest')
    logging.getLogger('take_photo').info(f"retrieved photo with shape: {img.shape}.")
    robot.camera.stop_video_stream()  # camera.take_photo doesn't seem to work so we start and stop video stream instead. 
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    photo_path = f"{photo_dir}/robot_photo_{timestamp}.jpg"
    if not cv2.imwrite(photo_path,img):
        raise Exception("Could not save image from robot. Check path or file extension.")
    else:
        logging.getLogger('take_photo').info(f"photo saved to {photo_path}")
    return img

def identify_speakers(audio_dir: str, speakerid_service):
    '''Use your speaker id model to identify speaker of each audio file in audio_dir
    Returns a dict that maps filename to speakerid.
    '''
    audio_dict = load_audio_from_dir(audio_dir)
    
    speakerid_result = {}
    for fname, v in audio_dict.items():
        audio_waveform, rate = v
        speakerid_result[fname] = speakerid_service.identify_speaker(audio_waveform, rate)
    
    for fname, speaker_id in speakerid_result.items():
        logging.getLogger('SpeakerID').info(f"{fname} speaker is {speaker_id}.")
    return speakerid_result

def detect_digits(digit_detection_service, audio_dir: str):
    digits_result = {}
    audio_dict = load_audio_from_dir(audio_dir)
    sorted_fnames = sorted(audio_dict.keys())

    for fname in sorted_fnames:
        audio_waveform, rate = audio_dict[fname]
        digits = digit_detection_service.transcribe_audio_to_digits(audio_waveform)
        digits_result[fname] = digits[0] if digits else None  # as a heuristic, take the first digit detected.
    return digits_result

def get_pose(loc_service, pose_filter):
    pose = loc_service.get_pose()

    if not pose:
        # no new pose data, continue to next iteration.
        return None

    return pose_filter.update(pose)

def plan_path(planner, start: list, goal):
    current_coord = RealLocation(x=start[0], y=start[1])
    path = planner.plan(current_coord, goal)
    return path

def ang_difference(ang1, ang2):
    '''Get angular difference in degrees of two angles in degrees, 
    
    Returns a value in the range [-180, 180]. 
    '''
    ang_diff = -(ang1 - ang2) # body frame

    # ensure ang_diff is in [-180, 180]
    if ang_diff < -180:
        ang_diff += 360

    if ang_diff > 180:
        ang_diff -= 360
    return ang_diff


def ang_diff_to_wp(pose, curr_wp):
    ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
    ang_diff = ang_difference(ang_to_wp, pose[2])
    return ang_diff

# Implement core robot loop here.
def main():
    def rotate_robot(last_valid_pose, angular_displacement):
        # ROTATE ROBOT BY angle DEGREES
    
        angle = (angular_displacement + last_valid_pose[2])
    
        while angle < -180:
            angle += 360
    
        while angle > 180:
            angle -= 360
    
        print("GO FROM", last_valid_pose[2], "TO", angle, "WITH ANGULAR DISPLACEMENT", angular_displacement)
    
        rel_ang = ang_difference(last_valid_pose[2], angle)  # current heading vs target heading
        logging.getLogger('Navigation').info("Turning robot to face target angle...")
        while abs(rel_ang) > 20:
            pose = get_pose(loc_service, pose_filter)
            rel_ang = ang_difference(pose[2], angle)  # current heading vs target heading
        
            if rel_ang < -20:
                # rotate counter-clockwise
                logging.getLogger('Navigation').info(f"Trying to turn clockwise... ang left: {rel_ang}")
                robot.chassis.drive_speed(x=0, z=10)
            elif rel_ang > 20:
                # rotate clockwise
                logging.getLogger('Navigation').info(f"Trying to turn counter-clockwise... ang left: {rel_ang}")
                robot.chassis.drive_speed(x=0, z=-10)
            time.sleep(0.1)
        robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0) 


    # === Initialize admin services ===
    loc_service = LocalizationService(host='localhost', port=5566)  # if passthrough sim, use sim's ip address.
    rep_service = ReportingService(host='localhost', port=5501)
    
    # === Initialize AI services. ===
    if cfg['use_real_models'] == True:
        reid_service = ObjectReIDService(yolo_model_path=CV_MODEL_DIR, reid_model_path=REID_MODEL_DIR)
        speakerid_service = SpeakerIDService(model_dir=SPEAKER_ID_MODEL_DIR)
        digit_detection_service = DigitDetectionService(model_dir=NLP_MODEL_DIR)
    else: # initialize mock AI classes.
        reid_service = MockObjectReIDService(yolo_model_path=CV_MODEL_DIR, reid_model_path=REID_MODEL_DIR)
        speakerid_service = MockSpeakerIDService(model_dir=SPEAKER_ID_MODEL_DIR)
        digit_detection_service = MockDigitDetectionService(model_dir=NLP_MODEL_DIR)
    
    # === Initialize robot ===
    robot = Robot()
    robot.initialize(conn_type="ap")
    robot.set_robot_mode(mode="chassis_lead")
    
    # === Initialize planner ===
    map_:SignedDistanceGrid = loc_service.get_map()
    map_ = map_.dilated(ROBOT_RADIUS_M) # dilate obstacles virtually so that planner avoids
                                        # bringing robot too close to real obstacles.
    planner = Planner(map_, sdf_weight=0.5)

    # === Initialize movement controller === 
    controller = PIDController(Kp=(0.7, 0.7), Ki=(0.3, 0.3), Kd=(0.3,0.3))  # this can be tuned.
    
    # === Initialize pose filter to smooth out noisy pose data === 
    pose_filter = SimpleMovingAverage(n=3)  # Smoothens out noisy localization data.
    
    # === Initialize your own variables ===
    curr_loi : RealLocation = None  # Current location of interest.
    path : List[RealLocation] = []  # list of way points to get from starting location to location of interest.
    curr_wp : RealLocation = None
    new_loi = None  # new RealLocation received from Reporting Server.
    target_rotation = None  # a bearing between [-180, 180]
    prev_loi = new_loi
    
    try_start_tasks = False
    
    # Tell reporting server we are ready.
    res = rep_service.start_run()
    if res.status == 200:
        initial_target_pose = eval(res.data)
        new_loi = RealLocation(x=initial_target_pose[0], y=initial_target_pose[1])
        target_rotation = initial_target_pose[2]
    else:
        logging.getLogger('Main').error("Bad response from challenge server.")
        return
    

    for _ in range(10):
        logging.getLogger('Main').info(f"Warming up pose filter to reduce initial noise.")
        pose = loc_service.get_pose()  # TODO: remove `clues`.
        time.sleep(0.1)
        pose = pose_filter.update(pose)
    
    logging.getLogger('Main').info(f">>>>> Autobot rolling out! <<<<<")

    # Main loop
    while True:
        pose = get_pose(loc_service, pose_filter)
        if pose is None:
            continue
        real_location = RealLocation(x=pose[0], y=pose[1])
        grid_location = map_.real_to_grid(real_location)
        if map_.in_bounds(grid_location) and map_.passable(grid_location):
            last_valid_pose = pose
        else:
            logging.getLogger('Main').warning(f"Invalid pose received from localization server.")
            continue
        
        if try_start_tasks:
            ## Check whether robot's pose is a checkpoint or not.
            info = rep_service.check_pose(pose)
            if type(info) == str:
                if info == "End Goal Reached":
                    rep_service.end_run()
                    print("=== YOU REACHED THE END ===")
                    break
                elif info == "Task Checkpoint Reached":
                        pass
                elif info == "Not An Expected Checkpoint" or info == "You Still Have Checkpoints":
                    logging.getLogger('Main').info(f"Not yet at task checkpoint. status: {res.status}, data: {res.data}, curr pose: {pose}")
                    # If we reached this execution branch, it means the autonomy code thinks the
                    # robot has reached close enough to the checkpoint, but the Reporting server
                    # is expecting the robot to be even closer to the checkpoint.
                    # TODO:  Robot should try to get closer to the checkpoint.
                    try_start_tasks = False
                    new_loi = prev_loi  # Try to navigate to prev loi again. May need to get bot to do
                                        # a pre-programmed sequence (e.g. shifting around slowly until its
                                        # position is close enough to the checkpoint).
                else:
                    raise Exception("Unexpected string value.")
            elif type(info) == RealPose:  # robot reached detour checkpoint and received new coordinates to go to.
                logging.getLogger('Main').info(f"Not goal, not task checkpt. Received a new target pose: {info}.")
                
                new_loi = RealLocation(x=info[0], y=info[1])
                target_rotation = info[2]
                logging.getLogger('Main').info(f"Setting {new_loi} as new LOI.")
                try_start_tasks = False
            else:
                raise Exception(f"Unexpected return type: {type(info)}.")

            if try_start_tasks:
                logging.getLogger('Main').info("===== Starting AI tasks =====")

                robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0) 
                time.sleep(0.1)
            
                # === OBJECT REIDENTIFICATION - Friend or Foe (Visual) ===
                # if reach checkpoint for first time, curr task is object re-id
    
                # TAKE PHOTO 

                has_sus = False
                has_hostage = False
                
                # rotate the camera by 90 degrees counterclockwise
                rotate_robot(last_valid_pose, -90)
                time.sleep(0.1)
                
                # update last_valid_pose
                while True:
                    pose = get_pose(loc_service, pose_filter)
                    if pose is None:
                        continue
                    real_location = RealLocation(x=pose[0], y=pose[1])
                    grid_location = map_.real_to_grid(real_location)
                    if map_.in_bounds(grid_location) and map_.passable(grid_location):
                        last_valid_pose = pose
                    else:
                        logging.getLogger('Main').warning(f"Invalid pose received from localization server.")
                        continue
                    break

                for reps in range(5):
                    print('\nRobot taking photo...')
                    img = take_photo(robot, PHOTO_DIR)
                    
                    sus_bbox, hostage_bbox = reid(reid_service, hostage_img, suspect_img, img)
                    # pass images to reid model to identify if scene contains hostage or suspect.
                
                    # Report to scoring server on who was found.
                    # NOTE: The scoring server implementation assumes that the suspect and hostage 
                    # will not be in same scene together. If they inadvertently are, then report the
                    # closer bounding box.

                    if sus_bbox:
                        has_sus = True
                    elif hostage_bbox:
                        has_hostage = True
                    
                    if has_sus or has_hostage:
                        break
                    
                    # rotate the camera by 45 degrees clockwise
                    rotate_robot(last_valid_pose, 45)
                    time.sleep(0.1)
                    # update last_valid_pose
                    while True:
                        pose = get_pose(loc_service, pose_filter)
                        if pose is None:
                            continue
                        real_location = RealLocation(x=pose[0], y=pose[1])
                        grid_location = map_.real_to_grid(real_location)
                        if map_.in_bounds(grid_location) and map_.passable(grid_location):
                            last_valid_pose = pose
                        else:
                            logging.getLogger('Main').warning(f"Invalid pose received from localization server.")
                            continue
                        break
                    
                if has_sus:
                    save_path = rep_service.report_situation(img, pose, "suspect", Path(ZIP_SAVE_DIR))
                elif has_hostage:
                    save_path = rep_service.report_situation(img, pose, "hostage", Path(ZIP_SAVE_DIR))
                else:
                    save_path = rep_service.report_situation(img, pose, "none", Path(ZIP_SAVE_DIR))
                
                logging.getLogger('Main').info(f"saved received files into {save_path}")
                
                ## === SPEAKER IDENTIFICATION - Friend or Foe (Audio) ===
                
                ## identify all speakers in a given directory
                speakerid_result = identify_speakers(save_path, speakerid_service)
                
                speakerid_submission = None
                for audio_fname, speakerid in speakerid_result.items():
                    if audio_fname.endswith('.wav'):
                        curr_audio_fname = audio_fname[:-4]
                        team_and_member = speakerid.split("_")
                        team = team_and_member[0]
                        if team.lower() == 'megumin':  # find opponent's clip.
                            speakerid_submission = curr_audio_fname + '_' + speakerid
                            break
                
                if speakerid_submission is None:
                    speakerid_submission = 'audio1_MEGUMIN_memberA'

                logging.getLogger('Main').info(f"submitting {speakerid_submission} to report_audio API.")
                save_path = rep_service.report_audio(pose, speakerid_submission, Path(ZIP_SAVE_DIR))
                logging.getLogger('Main').info(f"saved received files into {save_path}")

                # === DECODING DIGITS ===
                digits_result = detect_digits(digit_detection_service, save_path)
                password = tuple([val for _, val in digits_result.items()])
                
                # submit answer to scoring server and get scoring server's response.
                logging.getLogger('Main').info(f"Submitting password {password} to report_digit API.")
                target_pose = rep_service.report_digit(pose, password)
                logging.getLogger('Main').info(f"new target pose received from server: {target_pose}")
                
                new_loi = RealLocation(x=target_pose[0], y=target_pose[1])
                target_rotation = target_pose[2]
                    
                try_start_tasks = False
                logging.getLogger('Main').info("===== Ending AI tasks. Continuing Navigation to new target pose ======")

        # Path planning.
        curr_loi = new_loi
        curr_wp = None
        
        print("GOING TO", curr_loi)

        try:
            path = plan_path(planner, last_valid_pose, curr_loi)  ## Ensure only valid start positions are passed to the planner.
        except InvalidStartException as e:
            logging.getLogger('Navigation').warn(f"{e}")
            # TODO: find and use another valid start point.
            return
        logging.getLogger('Main').info('Path planned.')
        # TODO: abstract the path planning and movement code to functions.
                
        # Navigation loop.
        while (True):
            
            pose = get_pose(loc_service, pose_filter)
            if pose is None:
                continue
            
            real_location = RealLocation(x=pose[0], y=pose[1])
            grid_location = map_.real_to_grid(real_location)
            if map_.in_bounds(grid_location) and map_.passable(grid_location):
                last_valid_pose = pose
            else:
                logging.getLogger('Main').warning(f"Invalid pose received from localization server. Skipping.")
                continue
                
            dist_to_goal = euclidean_distance(last_valid_pose, curr_loi)
            if round(dist_to_goal, 2) <= REACHED_THRESHOLD_M:  # Reached checkpoint.
                logging.getLogger('Navigation').info(f"Reached checkpoint {last_valid_pose[0]:.2f},{last_valid_pose[1]:.2f}")
                path = []  # flush path.
                controller.reset()
                
                # ROTATE ROBOT TO TARGET ORIENTATION.
                rel_ang = ang_difference(last_valid_pose[2], target_rotation)  # current heading vs target heading
                logging.getLogger('Navigation').info("Turning robot to face target angle...")
                while abs(rel_ang) > 20:
                    pose = get_pose(loc_service, pose_filter)
                    rel_ang = ang_difference(pose[2], target_rotation)  # current heading vs target heading
                    
                    if rel_ang < -20:
                        # rotate counter-clockwise
                        logging.getLogger('Navigation').info(f"Trying to turn clockwise... ang left: {rel_ang}")
                        robot.chassis.drive_speed(x=0, z=10)
                    elif rel_ang > 20:
                        # rotate clockwise
                        logging.getLogger('Navigation').info(f"Trying to turn counter-clockwise... ang left: {rel_ang}")
                        robot.chassis.drive_speed(x=0, z=-10)
                    time.sleep(0.1)
                logging.getLogger('Navigation').info("Robot should now be facing close to target angle.")
                
                curr_wp = None
                prev_loi = curr_loi
                curr_loi = None
                try_start_tasks = True  # try to start ai tasks in the next main iter.
                break
            elif path:
                curr_wp = path[0]  # nearest waypoint is at the start of list.
                                
                logging.getLogger('Navigation').info(f"Num wps left: {len(path)}")
                
                dist_to_wp = euclidean_distance(real_location, curr_wp)
                if round(dist_to_wp, 2) < REACHED_THRESHOLD_M:
                    path = path[1:] # remove the nearest waypoint.
                    controller.reset()
                    curr_wp = path[0]
                    continue  # start navigating to next waypoint.         
                
                pose_str = f"x:{last_valid_pose[0]:.2f}, y:{last_valid_pose[1]:.2f}, rot:{last_valid_pose[2]:.2f}"
                curr_wp_str = f"{curr_wp[0]:.2f}, {curr_wp[1]:.2f}"
                curr_loi_str = f"{curr_loi[0]:.2f}, {curr_loi[1]:.2f}"
                dist_to_wp = euclidean_distance(real_location, curr_wp)
                logging.getLogger('Navigation').info(f"Goal: {curr_loi_str}, {target_rotation} \t Pose: {pose_str}")
                logging.getLogger('Navigation').info(f"WP: {curr_wp_str} \t dist_to_wp: {dist_to_wp:.2f}\n")
                
                ang_diff = ang_diff_to_wp(last_valid_pose, curr_wp)
                
                # Move robot until next waypoint is reached.
                # Determine velocity commands given distance to waypoint and heading to waypoint.
                vel_cmd = controller.update((dist_to_wp, ang_diff))
                vel_cmd[0] *= np.cos(np.radians(ang_diff)) # reduce forward velocity based on angular difference.
                
                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                forward_vel, ang_vel = vel_cmd[0], vel_cmd[1]
                logging.getLogger('Control').info(
                    f"input for final forward speed and rotation speed: {forward_vel:.2f}, {ang_vel:.2f}")
                
                robot.chassis.drive_speed(x=forward_vel, z=ang_vel)
                time.sleep(0.1)
            else:
                logging.getLogger('Navigation').info('Did not reach checkpoint and no waypoints left.')
                raise Exception('Did not reach checkpoint and no waypoints left.')

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.5)  # set stop for safety
    logging.getLogger('Main').info('===== Mission Terminated =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomy implementation')
    parser.add_argument('--config', type=str, help='path to configuration YAML file.', 
                        default="config/autonomy_cfg.yml")
    # As a best practice, use a configuration file to configure ur software.

    args = parser.parse_args()

    cfg_path = args.config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
        if cfg['use_real_models'] == True:
            from digit_detect_service import DigitDetectionService
            from speaker_id_service import SpeakerIDService
            from object_reid_service import ObjectReIDService
        else:
            from mock_ai_services import SiameseNetwork, MockDigitDetectionService, MockSpeakerIDService, MockObjectReIDService
            
        if cfg['use_real_localization'] == True:
            import robomaster
            from robomaster.robot import Robot  # Use this for real robot.
        else:
            from tilsdk.mock_robomaster.robot import Robot  # Use this for simulated robot.
        
        REACHED_THRESHOLD_M = cfg['REACHED_THRESHOLD_M']
        ANGLE_THRESHOLD_DEG = cfg['ANGLE_THRESHOLD_DEG']
        ROBOT_RADIUS_M = cfg['ROBOT_RADIUS_M']
        NLP_MODEL_DIR =  cfg['NLP_MODEL_DIR']
        CV_MODEL_DIR = cfg['CV_MODEL_DIR']
        REID_MODEL_DIR = cfg['REID_MODEL_DIR']
        SPEAKER_ID_MODEL_DIR = cfg['SPEAKER_ID_MODEL_DIR']
        PHOTO_DIR = cfg['PHOTO_DIR']

        STUCK_THRESHOLD = cfg['STUCK_THRESHOLD']  # number of iterations to consider robot as stuck on one waypoint.

        ZIP_SAVE_DIR = cfg ['ZIP_SAVE_DIR']

        # Initialize my team name.
        MY_TEAM_NAME = cfg['MY_TEAM_NAME']
        
        # Initialize target images of suspect and target hostage.
        suspect_img = cv2.imread(cfg['SUSPECT_IMG'])
        hostage_img = cv2.imread(cfg['HOSTAGE_IMG'])
        
    main()
