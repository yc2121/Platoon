## Simulation codes for the manuscript "Deep Reinforcement Learning for Multi-Objective Resource Allocation in Multi-Platoon Cooperative Vehicular Networks"

***

### Requirements to run the simulation program

> The simulation program requires Python3 with installed packages such as Pytorch, Numpy, Seaborn and etc.

### Structure of the simulation program

> `./src/algo` the folder of DRL algorithms.
> `./src/common` the folder of basic simulation settings, such as hyper parameters, saving path for models and log files and etc.
> `./src/env/Env_platoon.py` the simulator of the platoon communication environment.
> `./src/env/V2I.py` the simulator of the Vehicle-to-Infrastructure link.
> `./src/env/V2V.py` the simulator of the Vehicle-to-Vehicle link.
> `./src/env/vehicle.py` the simulator of the vehicle.
> `./src/neural_network/agent.py` the DRL agent for each platoon.
> `./src/neural_network/model.py` the neural network structure of the DRL algorithm.
> `./src/tools/plot_reward.py` the file to record the performance of different algorithms.
> `./src/tools/result_record.py` the file to record the training process and results.
> `./src/main.py` the file to start the training process.
> `./src/plot.py` the file to plot the figure of the performance of different algorithms.

### How to use the simulation program

- Run `./src/main.py` to start the training.
- After running the simulation program, the log information will be saved as log files such as `XXX.log` in the folder `logs`, the performance scores of DRL agents will be saved as txt files such as `XXX.txt` in the folder `data`.
- Run the `./src/plot.py` to plot the figure of the performance of different algorithms, the figure will be saved as pdf files such as `XXX.pdf` in the folder `data`.
