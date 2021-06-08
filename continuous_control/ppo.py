import json
import time
import numpy as np

import torch
import torch.optim as optim

from pathlib import Path
from collections import deque

from continuous_control.agent import Agent
from continuous_control.model import GaussianMLP


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
        self.env.reset(train=True)
        
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

            # Clip all actions to env action space range
            action = np.clip(action, -1, 1)

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

    def train(self, n_episodes=300, tmax=100, **kwargs):
        """
        Train agent

        Args:
            n_episodes (int): Number of training episodes
            tmax (int): Number of trajectories to collect

        Returns:
            None

        """
        scores = []                                 # scores from each episode
        best_avg_score = -100                       # best averaged window score
        best_avg_score_std = None                   # best averaged window score std
        score_goal = kwargs.get('goal', 30.0)       # goal to get to
        window_size = kwargs.get('window', 100)     # size for rolling window
        scores_window = deque(maxlen=window_size)   # last 100 scores

        # Init logging parameters
        run_id = kwargs.get('run_id', int(time.time()))
        output = kwargs.get('output', Path('./output/' + str(run_id) + '/'))
        verbose = kwargs.get('verbose', False)

        # Create log name
        prefix = str(run_id) + '__' + self.name + '__' + self.env.name
        log = output / (prefix + '__performance.log')

        # Iterate through episodes
        for episode in range(n_episodes):

            # Collect trajectories
            old_probs, states, actions, rewards = self._collect_trajectories(tmax=tmax)
            
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
            total_rewards = np.sum(rewards, axis=0)
            score = np.mean(total_rewards)
            scores.append(score)

            # Save most recent scores
            scores_window.append(score)
            scores.append(score)
            
            # Calculate average & standard deviation of current scores
            scores_mean = np.mean(scores_window)
            scores_std = np.std(scores_window)

            # Print & log episode performance
            window_summary = '\rEpisode {}\tAverage Score: {:.2f} ± {:.2f}'.format(episode, scores_mean, scores_std)
            print(window_summary, end="")

            # Check terminal condition every window_size episodes
            if episode % window_size == 0:
                
                # Save best performing model (weights)
                if scores_mean >= best_avg_score:
                    output.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), output / (prefix + '__best_model.pth'))
                    best_avg_score = scores_mean
                    best_avg_score_std = scores_std

                # Print & log performance of last window_size runs
                window_summary = '\rEpisode {}\tAverage Score: {:.2f} ± {:.2f}'.format(episode, scores_mean, scores_std)
                print(window_summary)
                self.write2path(window_summary, log)

                # Terminal condition check (early stop / overfitting)
                if scores_mean < best_avg_score:
                    window_summary = ('\rEarly stop at {:d}/{:d} episodes!\rAverage Score: {:.2f} ± {:.2f}'
                        '\tBest Average Score: {:.2f}').format(episode, n_episodes, scores_mean, scores_std, best_avg_score)
                    print(window_summary)
                    self.write2path(window_summary, log)
                    break

                # Terminal condition check (hit goal)
                if scores_mean - scores_std >= score_goal:
                    window_summary = '\nEnvironment solved in {:d}/{:d} episodes!\tAverage Score: {:.2f}±{:.2f}'.format(
                        episode, n_episodes, scores_mean, scores_std)
                    print(window_summary)
                    self.write2path(window_summary, log)
                    break

        # Save final model (weights)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), output / (prefix + '__model.pth'))
        
        # Plot training performance
        self.plot_performance(scores, output / (prefix + '__training.png'), window_size)
        
        # Save evaluation parameters
        parameters = {
            'n_episodes': n_episodes,
            #'eval_type': eval_type, 
            'tmax': tmax,
            'agent_seed': self.rng_seed,
            'env_seed': self.env.rng_seed,
            'best_avg_score': best_avg_score,
            'best_avg_score_std': best_avg_score_std,
            'scores_mean': scores_mean,
            'scores_std': scores_std
        }
        with open(output / (prefix + '__parameters.json'), 'w') as file:
            json.dump(parameters, file, indent=4, sort_keys=True)

        return
    
