# mfasaid
This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.

# Setting up a development environment in Windows

## Install Anaconda
Download and install the latest version of [Anaconda](https://www.anaconda.com/download/).

## Install Git
Download and install the latest version of [Git](https://git-scm.com/).

## Clone the repository
1. Open the Git Bash shell. 
2. Navigate to the directory where you want the top level directory of the repository to reside. Do this by using the `cd` command in Bash. For example, if I want mfasaid to reside in `D:\py\mfasaid`, I will type `cd /d/py` in the Bash prompt.
3. Clone the repository. For mfasaid, the command will be `git clone https://github.com/mdomanski-usgs/mfasaid.git`.

## Create and activate a new Conda environment
The environment.yml file contains a list of packages that will create an environment in which mfasaid has been designed to run.
1. Open the command prompt.
2. Navigate to the directory that contains the repository.
3. Create a new environment by typing `conda env create -f environment.yml` in the command prompt.
4. Activate the environment by typing `activate said` in the command prompt.
You must activate the `said` environment when you run SAID Python modules from the command line. For more information, see [this page](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Install the package
Since you want to modify the package while being able to import components into modules in other packages, you'll do an ["editable" install](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs).
1. In the Bash prompt, navigate to the directory that contains the Git repository. Continuing with the above example, the command would be `cd mfasaid`. Your current working directory will be shown as /d/py/mfasaid.
2. Type the command `pip install -e .`.
