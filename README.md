# til-23-welovejingyu

Winning code for the Novice category of Brainhack 2023's TIL-AI event.

Models used:

Object Detection: YOLOv6

Re-identification: ResNet152 with Binary Classification

Speaker Identification: TitaNet-L

Automatic Speech Recognition: ContextNet

Model weights can be found at https://drive.google.com/drive/folders/1LGb0pCE1zKgzcZQTmP9bM_vuROoMTeQq?usp=sharing.

# til-final

SDK, simulator and documentation for TIL 2023 Robotics Challenge.

* ``src/``: SDK and simulator packages. YOU DO NOT NEED TO MODIFY THESE.
* ``config/``: sample configuration files for the scoring server, simulator and sample autonomy code. 
  You might want to copy these.
* ``data/``: sample audio and images for the AI tasks, sample maps used for.
  the path planner and simulator. Feel free to use these data samples.
* ``docs/``: documentation and Sphinx docs source. Build these with the instructions below to view API docs.
* ``stubs/``: Code stubs for participants to implement. It is recommended to develop your custom codes in this directory.

## Setup
**Step 1: Clone this git repo.**

It is highly recommended to install python dependencies in a virtual environment.

**Step 2: Activate virtual environment.**
```
pip install virtualenv  # you do not need to run this line if you are down at our physical venue.
virtualenv -p python3.8.10 venv  # create virtual python environment with specific python version.
source <NAME_OF_VENV>/bin/activate  # on Windows, "./<NAME_OF_VENV>/Scripts/activate" to activate virtual python environment.
```

**Step 3: Install the [RoboMaster SDK](https://robomaster-dev.readthedocs.io/en/latest/python_sdk/installs.html)**
```sh
pip install robomaster
```

Note that the required python version for the RoboMaster SDK is between 3.6.6 and 3.8.10

**Step 4: Install the custom challenge SDK**
```sh
pip install -r requirements.txt  # install third-party dependencies. use the provided requirements-win.txt if installing on Windows.
pip install .  # install the SDK from source. The '.' means install from current directory.
```

**Step 5: Install a python GUI backend.**

On Ubuntu, try this GUI backend:
```
sudo apt-get install python3-tk  # If you are at our venue you don't need to do this (it is already installed).
```

or if you're on Windows
```
pip install pyqt5
```


**Step 6: (Optional) Install any other dependencies your custom AI models require.**

For example, the sample AI models require these.
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
export PYTHONPATH='/your/path/to/til-23-finals/stubs/yolov7'  # on Windows, replace 'export' with 'set'
```

Note: `set` command sets an environment variable temporarily only. Use 'setx' to set a variable permanently.   

## Build the documentation

Much of the technical information needed to understand and run the robotics system is provided in the Sphinx-generated docs.
Be sure to generate and view them!

```sh
# install sphinx dependencies
pip install sphinx sphinx-autoapi sphinx-rtd-theme

# Generate html docs based on source folder.
sphinx-build -b html docs/source docs/build 
```

Access the html docs at `docs/build/index.html`.

## Quickstart Example with Simulator

Open a terminal, start the scoring server:

`til-scoring config/scoring_cfg.yml -o teamName1`

In another terminal, start the simulator:

`til-simulator -c config/sim_cfg.yml`

In another terminal, start your autonomy code:

`python stubs/autonomy_starter.py --config config/autonomy_cfg.yml`

Note: REMEMBER to CHANGE THE FILE/DIRECTORY PATHS in the config files (by default in the "config" folder) to your own directories. And to modify the config files as necessary.

You are free not to use any of the provided stubs or code, except for the tilsdk's ReportingService and LocalizationService, which are mandatory for you to interact with our challenge servers.

You can always use the `--help` option with these commands to view help messages. E.g.

`til-scoring --help`

## Start developing

To prevent your code from being overwritten by patches and code releases by the organizers, you should make your own copies of the config files in the `config/` directory. and also your own copy of the `stubs/autonomy_starter.py` if you intend on using the sample code.

## Tips / Common Issues

Make sure your source code points to all the correct file and directory paths.

Ensure you are pointing your LocalizationService and ReportingService to the CORRECT IP addresses and port numbers where the servers are actually located.

You are supposed to make various improvements to the autonomy code. You can think about improvements to the robot movements, the path-planning, or the usage of the path planner's output waypoints, or the management of noisy localization data, etc etc. 
