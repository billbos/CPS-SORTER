# CPS-SORTER

CPS-SORTER uses AsFault to generate virtual test scenarios for self-driving car software. It provides different experiments to test the performance of a machine learning model based approach for test selection.
The goal of using a model, is to identify unsafe scenarios before test execution allowing to reduce testing cost by avoiding safe scenarios. The detailed description of the motivation and results can be found in the [master thesis](https://github.com/billbos/Master-Thesis-CPS-SORTER).

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
## Testgenerator (AsFault)
AsFault is an open-source project that allows creating test scenarios by applying a
search-based procedural content generation approach. AsFault focuses on the lane-keeping
of vehicles, trying to generate test scenarios that lead to the most out of bounds cases. 
AsFault can further be used to replay test scenarios, instructions on how to use asfault can be found in the [repo](https://github.com/alessiogambi/AsFault.git).
## Simulator (BeamNG)
BeamNG is an open world vehicle simulator that can be used freely for research purposes. It
offers a highly accurate physics engine, which is required for a simulator, that is used in research.
BeamNG is using a soft-body physic engine, simulating every component of a vehicle in realtime using nodes and beams. This contrast most simulators, where often a top-down approach is
used, where the physics of the vehicle is specified in its entirety, and the behavior and properties
of the components are derived from it. The bottom-up approach used by BeamNG should enable
a more detailed and authentic simulation. We used BeamNG to run and evaluate all of our test
scenarios.

## Interface to interact with Simulator (BeamNGPy)
BeamNGpy is a Python interface for BeamNG. The interface allows us to generate scenarios and handle simulations programmatically. Additionally, you can attach a different
kind of sensors and collect data during the simulations. We used BeamNGpy to modify the
process of generating scenarios in AsFault to create more complex environments, such as
adding weather effects.

# Experiments
We conducted three types of experiments  to run them you can use following commands.
## Evaluate Models
To evaluate different models on a dataset you can use 
```console
foo@CPS-SORTER:~$ cps_sorter run-model-eval -i /path/to/test/scenarios
```
### Mandatory parameters
* input directory (-i) define the location of the test scenarios

### Optional parameters
* datasetname (-d) define the name for the dataset evaluted, will be used in the result
* featureset (-f) either fullroad or roadsegment, defines what featureset should be extracted and used
* output dir (-o) define the location directory where the results will be persisted

## Evaluate Models
To evaluate different models on a dataset you can use 
```console
foo@CPS-SORTER:~$ cps_sorter run-model-eval -i /path/to/test/scenarios
```
### Mandatory parameters
* input directory (-i) define the location of the test scenarios

### Optional parameters
* rounds (-r) define the number of rounds conducted for the experiments
* ratio (-q) defines the ratio of the created dataset of unsafe and safe test scenarios, for the test pool
* output dir (-o) define the location directory where the results will be persisted

## Real-Time Experiments
To evaluate a pre-traineded adaptive model against baseline in a real time expeirment. To generate test scenarios finding as many unsafe as possible.
```console
foo@CPS-SORTER:~$ cps_sorter real-time-eval
```
### Optional parameters
* initial data (-i) path to a csv file with initial dataset that will be used for training
* timebudget (-t) defines the duration of the experiment
* output dir (-o) define the location directory where the results will be persisted
* adative (--adaptive/--no-adaptive) defines whether the model should continously trained with the newly generated data
