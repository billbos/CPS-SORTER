# CPS-SORTER

CPS-SORTER uses AsFault to generate virtual test scenarios for self-driving car software. It provides different experiments to test the performance of a machine learning model based approach for test selection.
The goal of using a model, is to identify unsafe scenarios before test execution allowing to reduce testing cost by avoiding safe scenarios.

# Structure

# Installation guide
This guide covers the installation of a local set-up of the project running on a single machine. It is recommended to install all components into a virtual environment in order to keep your Python installation as clean as possible.

### Requirements
* Python 3.6
* Windows (BeamNG runs only on windows)

## CPS-SORTER Installation
* Clone CPS-SORTER 
```console
foo@workspace:~$ git clone https://github.com/billbos/CPS-SORTER.git
```
* Install CPS-SORTER in virtual env
```console
foo@CPS-SORTER:~$ python -m venv venv
foo@CPS-SORTER:~$ ./venv/Scripts/activate
foo@CPS-SORTER:~$ pip install -r requirements.txt
foo@CPS-SORTER:~$ pip install -e .
```
## AsFault Installation
* Clone [AsFault](https://github.com/alessiogambi/AsFault.git) repo
```console
foo@workspace:~$ git clone https://github.com/alessiogambi/AsFault.git
```
* Use asfault-deap branch and install asfault
* Install asfault
* Install CPS-SORTER locally
```console
foo@CPS-SORTER:~$ pip install -e ./AsFault
```
## BeamNG research and PostgreSQL Installation
* This testing environment needs the ultimate version of the BeamNG research soft-body simulation. To get access you can contact the team behind [BeamNG research](https://beamng.gmbh/research/). Currently there is a version available for download [here](https://gwunipassaude-my.sharepoint.com/:u:/g/personal/huber176_gw_uni-passau_de/Ea5EMh0Ik9tKi43BY4HMPLsBMAHG_j74VsO4WGG3kHJnmQ?e=AGuRMj).
* Set environment variable BNG_HOME to point to the folder containing the executable of BeamNG.research. See instructions [here](https://superuser.com/q/949560).
For example, if you downloaded BeamNG.research to C:\Users\<You>Documents\beamng\research, where research is the name of the downloaded repository, then you should have the environment variable BNG_HOME that points to this path.

# Components

# Experiments
