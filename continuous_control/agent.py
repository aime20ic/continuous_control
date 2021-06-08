import json
import time
import numpy as np

import torch
import torch.optim as optim

from pathlib import Path

from continuous_control.model import GaussianMLP


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
        self.lr = kwargs.get('lr', 1e-4)                        # learning rate 
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

        # Agent properties
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.beta = kwargs.get('beta', 0.01)
        self.sgd_epoch = kwargs.get('sgd_epoch', 4)
        self.policy = GaussianMLP(env.state_size, env.action_size, 
            hidden=[128, 64], seed=seed).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        return

    def act(state):
        """
        """
        pass

    def _surrogate(self, old_probs, states, actions, rewards):
        """
        Clipped surrogate function for performing PPO. Modified from PPO lesson
        solution

        Args:
            old_probs (list): Action probabilities (mean & std)
            states (list): Trajectory states
            actions (list): Trajectory actions
            rewards (list): Trajectory rewards

        Returns:
            Policy loss

        """

        # Calculate discounted rewards for all envs wrt each time step
        discount = self.gamma**np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:,np.newaxis]
        
        # Calculate future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        
        # Normalize rewards
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis]) / std[:,np.newaxis]
        
        # Convert variables to tensors
        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int8, device=self.device)
        states = torch.stack(states)
        rewards_normalized = torch.tensor(rewards_normalized, dtype=torch.float, device=self.device)
        
        # Pass states to policy
        policy_dict = self.policy(states)
        new_probs = policy_dict['log_pi']
        entropy = -policy_dict['entropy']
            
        # Compute weights ratio
        ratio = new_probs / old_probs
            
        # clipped function
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        clipped_surrogate = torch.min(ratio*rewards_normalized, clip*rewards_normalized)
        
        return torch.mean(clipped_surrogate + self.beta * entropy)

    def _collect_trajectories(self, tmax=200, n_rand=10):
        """
        Collect trajectories from specified env after taking n_rand actions. 
        Modified from provided PPO Pong Utils 

        Args:
            tmax (int): Maximum number of time steps to collect
            n_rand (int): Number of random steps to perform before collecting 
                          trajectories
        
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
        
        # Perform n_rand random steps
        for _ in range(n_rand):
            _, _, _, state, _ = self.env.step(self.rng.uniform(-1,1,(n,4)))
        
        for t in range(tmax):

            # Prepare input
            state = torch.from_numpy(state).float().to(self.device)
            
            # Probs will only be used as the pi_old. No gradient propagation is 
            # needed, so we move it to the cpu
            policy_dict = self.policy(state)
            action = policy_dict['action'].squeeze().cpu().detach().numpy()
            probs = policy_dict['log_pi'].squeeze().cpu().detach().numpy()
                        
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

    def train(self, n_episodes=300, tmax=100):
        """
        Train agent

        Args:
            n_episodes (int): Number of training episodes
            tmax (int): Number of trajectories to collect

        Returns:
            None

        """

        # Keep track of progress
        mean_rewards = []

        # Iterate through episodes
        for e in range(n_episodes):

            # Collect trajectories
            old_probs, states, actions, rewards = self._collect_trajectories(tmax=tmax)
            
            # Aggregate rewards
            total_rewards = np.sum(rewards, axis=0)

            # Gradient ascent step
            for _ in range(self.sgd_epoch):
                
                # Get effective policy loss
                L = -self._surrogate(old_probs, states, actions, rewards)

                # Train / backprop
                self.optimizer.zero_grad()
                L.backward()
                self.optimizer.step()
                del L
            
            # Reduce clipping parameter as time goes on
            self.epsilon*=.999
            
            # Reduce regulation term to reduces exploration in later runs
            self.beta*=.995
            
            # Get average reward
            mean_rewards.append(np.mean(total_rewards))
            
            # Display some progress every 20 iterations
            if (e+1)%20 ==0 :
                print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
                print(total_rewards)
            
        return
    
