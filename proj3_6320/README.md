# Setup
1. Unzip proj3_6320.zip and go to proj3_6320 directory.
2. Install Miniconda. It doesn’t matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
3. Create a conda environment using the appropriate command. On Windows, open the installed “Conda prompt” to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (linux, mac, or win): conda env create -f proj3_env_<OS>.yml.
4. This should create an environment named ‘proj3’. Activate it using the Windows command, activate proj3 or the MacOS / Linux command, source activate proj3
5. Install the project package, by running pip install -e . inside the repo folder.
6. Run the notebook using jupyter notebook under proj3_6320 directory.
7. Ensure that all sanity checks are passing by running pytest tests inside the repo folder.

