Using the TIL SDK
=================

For the most part you will be importing and using classes provided from the ``tilsdk`` 
package to develop your code.

We also include these packages and their source code:

* ``tilsim``: the simulator
* ``tilscoring``: the scoring server

.. Note::
    You are not expected to modify any of the source codes provided in the ``tilsdk``, ``tilscoring``, 
    ``tilsim`` or packages. Import the relevant classes and functions and use them.

For explanations of the available APIs and how the packages are intended to be used, check out the 
`\to-\be-\released challenge website <https://github.com/til-23/til-23-info>`_ and these API docs :doc:`autoapi/index`.

You are *highly encouraged* to explore the API reference to understand how to use the core components and 
the optional (potentially helpful) components.


Using the provided SDK
----------------------

Once the TIL SDK is installed as per the instructions found in the root readme, 
the SDK can be used by importing modules from the ``tilsdk`` package.
e.g.

.. code-block:: python

    from tilsdk.mock_robomaster.robot import Robot                  # Use this mock robot for the simulator
    from tilsdk.utilities import PIDController, SimpleMovingAverage # import optional useful things
    from tilsdk.reporting import save_zip                           # helper function to handle embedded zip file in flask response
    # you can now call these modules, classes and functions in the rest of your code.

These are the main directories you would be working in when developing your autonomous robot 
application on the provided SDK:

* ``config/``: SDK and Simulator sample configuration files. You should copy these files and modify the config values. you may want to gitignore the contents of this folder.
* ``data/``: Simulator sample data. You should copy and modify these for your own use. you may want to gitignore the contents of this folder.
* ``stubs/``: Code stubs. You should implement the empty stubs - fill it with your code/components for the robot and AI tasks.
