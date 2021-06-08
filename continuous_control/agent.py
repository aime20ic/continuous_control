import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque


class Agent():
    """
    Agent
    """

    def __init__(self, env, seed=None, verbose=False, **kwargs):
        """
        Initialize an Agent object

        Args:
            env (UnityEnv): Simulation environment
            seed (int): Seed
            verbose (bool): Verbosity

        Returns:
            Agent object
        
        """

        # Set agent class variables
        self.name = kwargs.get('name', 'agent')
        self.run_id = kwargs.get('run_id', int(time.time()))
        self.output = kwargs.get('output', 
            Path('./output/' + str(self.run_id) + '/'))
        self.rng = None
        self.rng_seed = None
        self.verbose = verbose

        # Set environment variables
        self.env = env
        
        # Set agent training hyperparameters
        self.batch_size = kwargs.get('batch_size', 64)          # minibatch size
        self.gamma = kwargs.get('gamma', 0.99)                  # discount factor
        self.lr = kwargs.get('lr', 5e-4)                        # learning rate 
        self.optimizer = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set seed
        self.seed(seed)

        # Log class parameters
        self._log_parameters()

        return

    def load(self, path):
        """
        Load weights specified in path into both local & target networks

        Args:
            path (Path): Saved model weights to load

        Returns:
            None
        
        """
        print('Loading model from {}'.format(path.name))
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))
        return
    
    def act(self, state):
        """
        Returns actions for given state as per current policy
        
        Args:
            state (array): Current state

        Returns:
            Actions (array)
        
        """
        pass

    def seed(self, seed=None):
        """
        Set seed for random number generation, sampling, & repeatibility
        
        Args:
            seed (int): Seed for random number generation

        Returns:
            None
        
        """

        # Error check
        if not isinstance(seed, int) and seed is not None:
            raise ValueError('Specified seed must be integer or None')

        # Set seed & random number generator
        self.rng_seed = seed
        self.rng = np.random.RandomState(seed)

        return

    def _log_parameters(self):
        """
        Log agent parameters as JSON

        Args:
            None
        
        Returns:
            None
        
        """

        # Create file path
        path = self.output / (str(self.run_id) + '__' + self.name + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)

        # Make sure parameters are JSON serializable
        parameters = vars(self).copy()
        for key, value in parameters.items():
            try:
                json.dumps(value)
            except TypeError:
                parameters[key] = str(value)
        
        # Save as JSON
        with open(path, 'w') as file:
            json.dump(parameters, file, indent=4, sort_keys=True)

        return

    @staticmethod
    def write2path(text, path):
        """
        Write text to path object

        Args:
            text (str): Text to log
            path (Path): Path object
        
        Returns:
            None
        
        """

        # Create path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write text to path
        if not path.exists():
            path.write_text(text)
        else:
            with path.open('a') as f:
                f.write(text)
        
        return

    def plot_performance(self, scores, name, window_size):
        """
        Plot summary of DQN performance on environment

        Args:
            scores (list of float): Score per simulation episode
            name (Path): Name for file
            window_size (int): Windowed average size

        """
        window_avg = []
        window_std = []
        window = deque(maxlen=window_size)
        
        # Create avg score
        avg = [np.mean(scores[:i+1]) for i in range(len(scores))]
        for i in range(len(scores)):
            window.append(scores[i])
            window_avg.append(np.mean(window))
            window_std.append(np.std(window))

        # Create 95% confidence interval (2*std)
        lower_95 = np.array(window_avg) - 2 * np.array(window_std)
        upper_95 = np.array(window_avg) + 2 * np.array(window_std)

        # Plot scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(np.arange(len(scores)), scores, color='cyan', label='Scores')
        plt.plot(avg, color='blue', label='Average')
        plt.plot(window_avg, color='red', label='Windowed Average (n={})'.format(window_size))
        plt.fill_between(np.arange(len(window_std)), lower_95, upper_95, color='red', alpha=0.1)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.legend(loc='upper left')
        ax.margins(y=.1, x=.1) # Help with scaling
        plt.show(block=False)

        # Save figure
        name.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(name)
        plt.close()

        return


