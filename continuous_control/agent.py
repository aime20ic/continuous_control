import json
import time
import numpy as np

import torch

from pathlib import Path

from continuous_control.model import MLP


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


class PPO(Agent):
    """
    Proximal Policy OPtimization Agent
    """

    def __init__(self, env, seed=None, verbose=False, **kwargs):
        """
        Initialize a PPO Agent object
        """

        # Agent initialization
        super().__init__(env, seed, verbose, **kwargs)

        # PPO specific
        self.policy = MLP(env.state_size, 2*env.action_size, 
            hidden=[128, 64], seed=seed).to(self.device)

        return

    def act(state):
        """
        """
        pass

    def _surrogate(policy, old_probs, states, actions, rewards, 
            discount=0.995, epsilon=0.1, beta=0.01):
        """
        Clipped surrogate function for performing PPO. Modified from PPO lesson
        solution

        Args:
            policy (Torch nn.Module): Torch neural network
            old_probs (list): Action probabilities
            states (list): Trajectory states
            actions (list): Trajectory actions
            rewards (list): Trajectory rewards
            discount (float): Reward discount factor
            epsilon (float): Gradient clip factor
            beta (float): Entropy scale factor

        Returns:
            Policy gradient

        """
    
        # Calculate discounted rewards for all envs wrt each time step
        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:,np.newaxis]
        
        # Calculate future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        
        # Normalize rewards
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis]) / std[:,np.newaxis]
        
        # Convert variables to tensors
        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        states = torch.stack(states)
        rewards_normalized = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
        
        # Convert states from tensor shape of 
        # [tmax, n_parallel_env, 2, 80, 80] -> [tmax * n_parallel_env, 2, 80, 80]
        policy_input = states.view(-1, *states.shape[-3:])
        
        # Pass states to policy
        new_probs = policy(policy_input).view(states.shape[:-3])
        
        # convert states to policy (or probability) wrt to action 
        # new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
        
        # Compute weights ratio
        # with torch.no_grad():
        ratio = new_probs / old_probs
            
        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards_normalized, clip*rewards_normalized)
        
        # include a regularization term
        # this steers new_policy towards 0.5
        # which prevents policy to become exactly 0 or 1
        # this helps with exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(
            new_probs * torch.log(old_probs + 1.e-10) + 
            (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10)
        )

        return torch.mean(clipped_surrogate + beta * entropy)

    def _get_actions(self, distributions, min_val=-1.0, max_val=1.0):
        """
        Sample actions from specified distribution (mean & std)

        Args: 
            distributions (list of float): Mean & standard deviation for 
                                           distribution per agent in env
            min_val (float): Minimum action value
            max_val (float): Maximum action value

        Returns:
            Samples from distributions

        """ 
        samples = []

        # Sample from distribution
        for agent in distributions:
            agent_samples = []

            for mean, std in agent.reshape(len(agent)//2, 2):

                # Std must be positive
                if std < 0:
                    std = 0.0

                # Get sample
                sample = self.rng.normal(mean, std)

                # Limit action to valid range
                if sample < min_val:
                    sample = min_val
                if sample > max_val:
                    sample = max_val

                # Append to list
                agent_samples.append(sample)
            
            # Append to sample list
            samples.append(agent_samples)

        return samples

    def _collect_trajectories(self, tmax=200, n_rand=5):
        """
        Collect trajectories from specified env after taking n_rand actions. 
        Modified from provided PPO Pong Utils 

        Args:
            tmax (int): Maximum number of time steps to collect
            n_rand (int):
        
        Returns:
            List of collected probabilities, states, actions, & rewards

        """

        # Number of parallel instances
        n = len(self.env.agents)

        # Initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        # Reset env
        self.env.reset()
        
        # Start all parallel agents
        # env.step([1]*n)
        
        # Perform n_rand random steps
        for _ in range(n_rand):
            _, _, _, state, _ = self.env.step(self.rng.uniform(-1,1,(n,4)))
        
        for t in range(tmax):

            # Prepare input
            state = torch.from_numpy(state).float().to(self.device)
            
            # Probs will only be used as the pi_old. No gradient propagation is 
            # needed, so we move it to the cpu
            probs = self.policy(state).squeeze().cpu().detach().numpy()

            # Get actions from action probabilities (mean, std)
            action = self._get_actions(probs)
            
            # Advance game using determined actions
            _, _, reward, next_state, done = self.env.step(action)
            
            # store the result
            state_list.append(state)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)

            # Set state
            state = next_state

            # Stop if any of the trajectories are done
            if any(done):
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list