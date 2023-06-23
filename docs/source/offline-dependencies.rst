Managing dependencies in an offline environment
===============================================


Saving your dependencies into a transferrable format
----------------------------------------------------

Assumptions:

- You have git pulled the til-23-finals repo and your current working directory 
  is the til-23-finals repo root
- you are in an internet-enabled environment

**Download all pypi-indexed dependencies for your code**

.. code-block:: shell

    pip download -r path/to/requirements.txt -d /path/to/downloaded/packages  
    # download all third-party dependencies from the pypi index by default

**Make dist wheel for til-23-finals**

.. code-block:: shell

    python setup.py bdist_wheel --universal
    # creates the wheel from the setup.py of the current python project.

**Transfer til-23-finals wheel file to dest folder**

copy-paste wheel file generated from prev step into target directory of packages.

**Transfer the folder of wheel files to a portable drive**

copy-paste folder of downloaded packages (wheel files) into a portable drive, e.g. a flash drive.


Installing from packages stored offline 
---------------------------------------

Now let's install from the dependency files stored in the portable drive.

Assumptions:

- You have activited a python virtual environment (recommended)

**Install til-23-finals dependencies and your own dependencies**

.. code-block:: shell

    pip install --no-index --find-links="/path/to/downloaded/packages" -r requirements.txt

**Install til-23-finals package**

.. code-block:: shell

    pip install --no-index --find-links="/path/to/downloaded/packages" til-23-finals

**Test that the installation of the til-23-finals package is succesful by running these commands**

.. code-block:: shell

    til-scoring --help

.. code-block:: shell

    til-simulator --help

If the installation was successful, you should see the help messages of these commands.
