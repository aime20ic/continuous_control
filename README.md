[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: ./images/agent_performance.png "DDPG Agent Performance"

# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

**NOTE: THIS IS THE OPTION THAT WAS SOLVED FOR THIS PROJECT**

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Installation

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - __Linux__ or __Mac__:
  
    ```bash
    conda create --name cont_ctrl python=3.6
    source activate cont_ctrl
    ```

   - __Windows__:

    ```bash
    conda create --name cont_ctrl python=3.6 
    activate cont_ctrl
    ```

2. Clone Navigation repository 

    ```bash
    git clone git@github.com:aime20ic/continuous_control.git
    ```

3. Install [dependencies](#dependencies)

4. Download [Unity Simulation Environment](#unity-simulation-environment)

### Dependencies

To install required dependencies to execute code in the repository, follow the instructions below.

1. Install [PyTorch](https://pytorch.org/)

    ```bash
    conda install pytorch cudatoolkit=10.2 -c pytorch
    ```

    A fresh installation of [PyTorch](https://pytorch.org/) is recommended due to installation errors encountered when installing the [Udacity Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning), such as [the following Windows error](https://github.com/udacity/deep-reinforcement-learning/issues/13), as well as outdated driver issues when using `torch==0.4`.

2. Install required packages using `pip` from main repository directory

    ```bash
    cd continuous_control
    pip install .
    ```

### Unity Simulation Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Save the file locally and unzip (or decompress) the file.

## Instructions

The `run_agent.py` script can be used to train or evaluate an agent. Logs of agent parameters, agent performance (as shown below), and environment evaluation settings are saved during script execution. Some agent and associated model hyperparameters are configurable via command line arguments. See [help](#help) section for more details about available parameters.

![DDPG Agent Performance][image2]

### Training an Agent

Training an agent only requires specifying the path to the [downloaded Unity simulation environment](#getting-started)

```bash
python -m continuous_control.run_agent --sim Reacher__Windows_x86_64/Reacher.exe
```

Model training parameters are configurable via command line. Certain variables such as env name are used for
logging of agent parameters and environment performance results.

```bash
python -m continuous_control.run_agent --sim Reacher__Windows_x86_64/Reacher.exe --n-episodes 1000 --env-name reacher-env --seed 5
```

### Continuing Training

To continue training using a previously trained model, specify the path to the previously saved model using the `--actor` and `--critic` command line arguments.

**NOTE:** Saved model hidden layer sizes must match sizes specified in `ddpg_model.py`. 

```bash
python -m continuous_control.run_agent --sim Reacher__Windows_x86_64/Reacher.exe --actor example_models/actor.pth --critic example_models/critic.pth
```

### Evaluating a Trained Agent

Evaluating a trained agent requires using the `--actor` and `--critic` command line arguments as well as `--test` argument simultaneously. The number of evaluation episodes is specified using `--n-episodes` argument, while `--max-t` argument specifies the number of maximum simulation time steps per episode. The maxmimum number of time steps for the Unity Reacher simulation appears to be 1000 (determined empirically).

```bash
python -m continuous_control.run_agent --sim Reacher__Windows_x86_64/Reacher.exe --actor example_models/actor.pth --critic example_models/critic.pth --n-episodes 100 --test --seed 15
```

### Help

For a full list of available parameters try

```bash
python -m continuous_control.run_agent --help

usage: run_agent.py [-h] [--actor ACTOR] [--critic CRITIC]
                    [--env-name ENV_NAME] [--goal GOAL] [--max-t MAX_T]
                    [--n-episodes N_EPISODES] [--output OUTPUT]
                    [--run-id RUN_ID] [--sim SIM] [--test] [--window WINDOW]
                    [--seed SEED] [--verbose]

Agent hyperparameters

optional arguments:
  -h, --help            show this help message and exit
  --actor ACTOR         Path to actor model to load
  --critic CRITIC       Path to actor model to load
  --env-name ENV_NAME   UnityEnv name
  --goal GOAL           Score goal
  --max-t MAX_T         Maximum number of timesteps per episode
  --n-episodes N_EPISODES
                        Maximum number of training episodes
  --output OUTPUT       Directory to save models, logs, & other output
  --run-id RUN_ID       Execution run identifier
  --sim SIM             Path to Unity Reacher simulation
  --test                Test mode, no agent training
  --window WINDOW       Window size to use for terminal condition check
  --seed SEED           Seed for repeatability
  --verbose             Verbosity
  ```
