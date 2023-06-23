import argparse
import asyncio
from enum import Enum
import io
import logging
import queue
import shelve
import os
import yaml
import zipfile
from datetime import datetime
from pathlib import Path

import flask
from flask_cors import CORS, cross_origin
from werkzeug.serving import WSGIRequestHandler

from .types import *
from .messenger import MessageAnnouncer

 
class CheckpointState(Enum):
    RUNNING = -1
    VALID_CHECKPOINT = 0
    CV_DONE = 1
    SPEAKER_DONE = 2

#### Settings #####
TIME_THRESHOLD_S = 1.0  # minimum delay in seconds


##### Global variables #####
out_dir:str = '.'
out_file:shelve.Shelf = None
start_time:datetime = None
last_recv_time:datetime = None
valid_pose = []
detour_pose = []
pose_counter = 0
checkpoint_state = CheckpointState.RUNNING
score = 0
total_score = 0

#### ArUco #####
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # use different from localization system
# aruco_params = cv2.aruco.DetectorParameters_create()


##### Flask server #####

# Flask defaults to HTTP 1.0. Use HTTP 1.1 to keep connections alive for high frequency requests.
WSGIRequestHandler.protocol_version = 'HTTP/1.1'

announcer = MessageAnnouncer()
# For pushing robot status notifications to listening clients.

app = flask.Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/listen', methods=['GET'])
@cross_origin(send_wildcard=True)
def listen():
    """
    Endpoint for clients to listen/subscribe to notifications about the robot's 
    challenge status.
    """
    logging.getLogger('listen').info(f"/listen called.")
    def stream():
        messages = announcer.listen()  # returns a queue.Queue
        while True:
            logging.getLogger('listen').info(f"in while loop to get messages...")    
            msg = messages.get()  # blocks until a new message arrives
            logging.getLogger('listen').info(f"message received:")
            yield msg
    res = flask.Response(stream(), mimetype='text/event-stream')
    #res.headers.add("Access-Control-Allow-Origin", "*")
    return res


@app.route('/start_run', methods=['GET'])
def get_start_run():
    
    global out_dir, out_file, start_time, last_recv_time, pose_counter
    global announcer, checkpoint_state, score, total_score
    global valid_pose, detour_pose

    if out_file:
        out_file.close()
        out_dir = '.'
        out_file = None
        start_time = None
        last_recv_time = None
        valid_pose = []
        detour_pose = []
        pose_counter = 0
        checkpoint_state = CheckpointState.RUNNING
        score = 0
        total_score = 0

    start_time = datetime.now()
    ts = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    fname = 'run_{}.shelve'.format(ts)
    out_fpath = os.path.join(out_dir, fname)
    out_file = shelve.open(out_fpath, protocol=4)
    logging.getLogger('start_run').info('Run started at {}. Writing to {}'.format(ts, out_fpath))
   
    # Read the config file for valid and detour checkpoints
    checkpoint_pose = config['valid_checkpoints']
    for pose in checkpoint_pose: 
        new_pose = eval(pose)
        new_pose = tuple(float(p) for p in new_pose)  # fix to float instead of int.
        valid_pose.append(RealPose(new_pose[0],new_pose[1],new_pose[2]))

    logging.getLogger('start_run').info('valid checkpoints : {}'.format(valid_pose))

    checkpoint_pose = config['detour_checkpoint']
    for pose in checkpoint_pose: 
        new_pose = eval(pose)
        new_pose = tuple(float(p) for p in new_pose)  # fix to float instead of int.
        detour_pose.append(RealPose(new_pose[0],new_pose[1],new_pose[2]))

    logging.getLogger('start_run').info('detour checkpoints : {}'.format(detour_pose))
    
    location = valid_pose[pose_counter]
    logging.getLogger('start_run').info('First checkpoint given {} with pose counter at {}'.format(location,pose_counter))
    
    team_name = os.path.split(out_dir)[-1]
    
    status_dict = {
        'status':  "start run",
        'time': start_time.isoformat(),
        'team_name': team_name
    }
    announcer.announce(event='status update', data_dict=status_dict)
    
    return str(location),200

@app.route('/end_run', methods=['GET'])
def get_end_run():
    
    global out_dir, out_file, start_time, total_score
    if out_file:
        end_time = datetime.now()
        ts = end_time.strftime('%H%M%S')
        fn = 'run_{}.shelve'.format(ts)
        report = Report()
        report.id = 'Ended'
        report.timestamp = end_time
        diff_time = end_time-start_time
        report.situation  = str(diff_time.total_seconds()) + ' s'
        report.score = total_score
        out_file[report.id] = report

        out_file.close()
        logging.getLogger('end_run').info('Run ended at {}'.format(ts))
        return 'OK',200

    else : 
        return 'The run was never started', 400

@app.route('/check_pose', methods=['GET'])
def get_check_pose():
    global last_recv_time, report, valid_pose, pose_counter, checkpoint_state, score, start_time, announcer

    recv_time = datetime.now()
    logging.getLogger('check_pose').info('check_pose received at {}'.format(recv_time.strftime('%H:%M:%S')))\

    try: 
        if (recv_time - start_time).total_seconds() > config['max_time_per_run_s']:
            logging.getLogger('check_pose').info('Run max time exceeded.')
            out_file.close()
            return 'Run time exceeded.', 400
    except TypeError:
        logging.getLogger('check_pose').info('start_run was never called')
        return 'Please start the run first by calling <start_run>',400


    content_type = flask.request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = flask.request.json
        pose = json['pose']

        new_pose=tuple(pose)
        pose = RealPose(new_pose[0],new_pose[1],new_pose[2])
        valid = valid_pose[pose_counter]
        detour = detour_pose[pose_counter]

        #print(f"euclid dist to valid {valid}: {euclidean_distance(pose, valid)}")
        #print(f"euclid dist to detour {detour}: {euclidean_distance(pose, detour)}")

        time_elapsed_s = (datetime.now() - start_time).seconds
        
        # if pose_counter > len(valid_pose):
        #     return 'Please exit the maze using the last location given',300
        
        if euclidean_distance(pose, valid_pose[-1]) < config['local_thres']:  # near end goal.
            if pose_counter < (len(valid_pose) - 1):
                return "You Still Have Checkpoints", 300
            else:
                checkpoint_state = CheckpointState.VALID_CHECKPOINT
                score=0

                status_dict = {
                    'status':  'goal reached',
                    'time_elapsed': time_elapsed_s
                }
                announcer.announce(event='status update', data_dict=status_dict)

                return 'End Goal Reached', 200
        elif euclidean_distance(pose, valid) < config['local_thres']:
            checkpoint_state = CheckpointState.VALID_CHECKPOINT
            
            status_dict = {
                'status':  'task checkpoint reached',
                'time_elapsed': time_elapsed_s,
                'checkpoint_number': pose_counter + 1
            }
            announcer.announce(event='status update', data_dict=status_dict)
            return 'Task Checkpoint Reached', 200

        elif euclidean_distance(pose, detour) < config['local_thres']:
            checkpoint_state = CheckpointState.RUNNING
            
            status_dict = {
                'status':  'detour checkpoint reached',
                'time_elapsed': time_elapsed_s,
                'checkpoint_number': pose_counter + 1
            }
            announcer.announce(event='status update', data_dict=status_dict)        

            # Reached detour checkpoint, so return pose of next task checkpoint.
            return str(valid_pose[pose_counter]), 200
        else: 
            checkpoint_state = CheckpointState.RUNNING
            return 'Not An Expected Checkpoint', 300

@app.route('/report_situation', methods=['POST'])
def post_report_situation():
    global last_recv_time, report, checkpoint_state, score, start_time, announcer 
    CORRECT_SCORE = 10
    points_given = 0
    
    recv_time = datetime.now()
    logging.getLogger('report_situation').info('Report Situtation received at {}'.format(recv_time.strftime('%H:%M:%S')))
    time_elapsed_s = (recv_time - start_time).seconds

    if checkpoint_state.value != 0: 
        logging.getLogger('report_situation').info('<check_pose> was not called before')
        return 'Please check whether you are in the correct checkpoint <check_pose> before reporting the situation.',400

    if (recv_time - start_time).total_seconds() > config['max_time_per_run_s']:
        logging.getLogger('report_situation').info('Run max time exceeded.')
        out_file.close()
        return 'Run time exceeded.', 400

    report = Report()
    # report.id=str(recv_time.timestamp())
    report.id = str(pose_counter)
    report.timestamp=recv_time

    if last_recv_time and (recv_time - last_recv_time).total_seconds() < TIME_THRESHOLD_S:
        logging.getLogger('report_situation').info('Report within time threshold, IGNORED.')
        report.time_valid = False
        out_file[report.id] = report
        return 'OK', 200
    else:
        report.time_valid = True
        last_recv_time = recv_time

        content_type = flask.request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json = flask.request.json
            pose = json['pose']
            report.checkpoint_id = pose_counter + 1  # checkpoint id is 1-based, pose_counter list index is 0-based.
            report.pose = pose

            report.image = json['image']
            report.situation = json['situation'].lower()

        actual_situation = config['targets'][pose_counter+1]['situation'].lower()
        if report.situation != actual_situation:
            logging.getLogger('report_situation').info('Wrong situation reported: received {} when the situation is {}'.format(report.situation,actual_situation))
            report.time_valid = False
        else: 
            score += CORRECT_SCORE
            points_given = CORRECT_SCORE
            logging.getLogger('report_situation').info('Correct situation reported: received {} when the situation is {}'.format(report.situation,actual_situation))

        file_path = config['targets'][pose_counter+1]['audio_path']
        file_path = Path(file_path).as_posix()
        logging.getLogger('report_situation').info('file_path {}'.format(file_path))

        checkpoint_state = CheckpointState.CV_DONE

        # Send the following fields for audience visualization.
        status_dict = {
            'status': 're-id done',
            'time_elapsed': time_elapsed_s,
            'submitted_image': report.image,
            'submitted_answer':  report.situation,
            'correct_answer': actual_situation,
            'points_given': points_given
        }
        announcer.announce(event='status update', data_dict=status_dict)
        
        #TODO : Handle filename to return 
        return flask.send_file(file_path, download_name='audio_files_identify.zip', as_attachment=True)

@app.route('/report_audio', methods=['POST'])
def post_report_audio():
    global last_recv_time, report, checkpoint_state, score, start_time
    CORRECT_SCORE = 5
    points_given = 0
    
    recv_time = datetime.now()
    logging.getLogger('report_audio').info('Report Audio received at {}'.format(recv_time.strftime('%H:%M:%S')))
    time_elapsed_s = (recv_time - start_time).seconds

    if checkpoint_state != CheckpointState.CV_DONE: 
        logging.getLogger('report_audio').info('Please run the previous APIs first')
        return 'Please run the previous APIs first',300
        
    if (recv_time - start_time).total_seconds() > config['max_time_per_run_s']:
        logging.getLogger('report_audio').info('Run max time exceeded.')
        out_file.close()
        return 'Run time exceeded.', 400

    last_recv_time = recv_time

    # parse results sent in to report about the situation 
    content_type = flask.request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = flask.request.json
        audio =  json['chosen_audio']
        report.audio = audio
    else:
        raise Exception(f"Unexpected request content type: {content_type}.")
    
    actual_audio = config['targets'][pose_counter+1]['correct_audio']
    if audio.lower() != actual_audio.lower():  # case-insensitive comparison.
        logging.getLogger('report_audio').info('Wrong audio reported: received {} when the expected audio is {}'.format(audio, actual_audio))
        report.time_valid = False
    else: 
        score += CORRECT_SCORE
        points_given = CORRECT_SCORE 
        logging.getLogger('report_audio').info('Correct audio reported: received {} when the expected audio is {}'.format(audio, actual_audio))

    file_path = config['targets'][pose_counter+1]['digit_audio_path']    
    logging.getLogger('report_audio').info('file_path {}'.format(file_path))
    
    checkpoint_state = CheckpointState.SPEAKER_DONE
    
    #TODO: send the following fields to the audience visualization.
    status_dict = {
        'status': 'speaker id done',
        'time_elapsed': time_elapsed_s,
        'submitted_answer':  report.audio,
        'correct_answer': actual_audio,
        'points_given': points_given
    }
    announcer.announce(event='status update', data_dict=status_dict)
    
    return flask.send_file(file_path, download_name='audio_files_digits.zip', as_attachment=True)


@app.route('/report_digit', methods=['POST'])
def post_report_digit():
    global start_time, last_recv_time, report, checkpoint_state, score, total_score, pose_counter, detour_pose, valid_pose, announcer
    CORRECT_SCORE = 0
    
    recv_time = datetime.now()
    logging.getLogger('report_digit').info('Report Digit received at {}'.format(recv_time.strftime('%H:%M:%S')))
    time_elapsed_s = (recv_time - start_time).seconds

    if checkpoint_state not in [CheckpointState.CV_DONE, CheckpointState.SPEAKER_DONE]:
        logging.getLogger('report_digit').info('Please run the previous APIs first')
        return 'Please run the previous APIs first', 300

    if (recv_time - start_time).total_seconds() > config['max_time_per_run_s']:
        logging.getLogger('report_digit').info('Run max time exceeded.')
        out_file.close()
        return 'Run time exceeded.', 400

    last_recv_time = recv_time

    # parse results sent in to report about the digits 
    content_type = flask.request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = flask.request.json
        pose = json['pose']
        digits = json['recover_digits']
        report.digits = tuple(digits) 

    actual_digits = eval(config['targets'][pose_counter+1]['digits'])
    if report.digits == actual_digits:
        score += CORRECT_SCORE
        points_given = CORRECT_SCORE
        location = valid_pose[pose_counter+1]
        logging.getLogger('report_digits').info('Correct digits reported: received {} but expected is {}. Correct pose {} returned'.format(report.digits,actual_digits,location))
    else:
        report.time_valid = False
        points_given = 0
        location = detour_pose[pose_counter+1]
        logging.getLogger('report_digits').info('Wrong digits reported: received {} but expected is {}. Detour pose {} returned'.format(report.digits,actual_digits,location))

    report.score = score
    total_score += score
    pose_counter += 1 
    score = 0
    out_file[report.id] = report

    checkpoint_state = CheckpointState.RUNNING

    status_dict = {
        'status': 'digits done',
        'time_elapsed': time_elapsed_s,
        'submitted_answer':  report.digits,
        'correct_answer': actual_digits,
        'points_given': points_given
    }
    announcer.announce(event='status update', data_dict=status_dict)

    return str(location),200


def validate(cfg: dict):
    checkpoints = cfg['targets']
    
    for checkpoint, params in checkpoints.items():
        print(f"checkpoint: {checkpoint}")
        print(f"values: {params}\n")
        
        assert params['situation'] in ['suspect', 'hostage', 'none']
        
        # Check zip files exist.
        assert os.path.exists(params['audio_path'])
        assert zipfile.is_zipfile(params['audio_path'])
        assert os.path.exists(params['digit_audio_path'])
        assert zipfile.is_zipfile(params['digit_audio_path'])
    print("passed assertions!")
        
    

def main():
    global config, out_dir

    parser = argparse.ArgumentParser(description='TIL Scoring Server.')
    parser.add_argument('config', type=str, help='Scoring configuration YAML file.')
    parser.add_argument('-i', '--host', metavar='host', type=str, required=False, default='0.0.0.0', help='Server hostname or IP address. (Default: "0.0.0.0")')
    parser.add_argument('-p', '--port', metavar='port', type=int, required=False, default=5501, help='Server port number. (Default: 5501)')
    parser.add_argument('-o', '--out_dir', dest='out_dir', type=str, required=False, default='./scoring', help='Scoring output directory.')
    parser.add_argument('-ll', '--log', dest='log_level', metavar='level', type=str, required=False, default='info', help='Logging level. (Default: "info")')
    args = parser.parse_args()

    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    ##### Setup logging #####
    map_log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    logging.basicConfig(level=map_log_level[args.log_level],
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(os.path.join(out_dir, 'scoring_log.txt'))
                    ])

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        validate(config)

    app.run(args.host, args.port)

if __name__ == '__main__':
    main()