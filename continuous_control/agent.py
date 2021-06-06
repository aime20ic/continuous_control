import json
import time
import numpy as np

from pathlib import Path

from continuous_control.model import MLP


class Agent():
    """
    Agent
    """

    def __init__(state_size, action_size, seed=None, verbose=False, **kwargs):
        """
        Initialize an Agent object
        """
        # Set agent class variables
        self.t_step = 0
        self.name = kwargs.get('name', 'agent')
        self.run_id = kwargs.get('run_id', int(time.time()))
        self.output = kwargs.get('output', 
            Path('./output/' + str(self.run_id) + '/'))
        self.rng = None
        self.rng_seed = None
        self.verbose = verbose

        # Set environment variables
        self.state_size = state_size
        self.action_size = action_size
        
        # Set agent training hyperparameters
        self.batch_size = kwargs.get('batch_size', 64)          # minibatch size
        self.gamma = kwargs.get('gamma', 0.99)                  # discount factor
        self.lr = kwargs.get('lr', 5e-4)                        # learning rate 
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

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
            None
        
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

