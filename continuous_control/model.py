import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi Layer Perceptron Model"""

    def __init__(self, state_size, action_size, hidden=[128,64], seed=None):
        """Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden (list of int): List specifying hidden layer sizes
        
        Returns:
            None
        
        """
        super(MLP, self).__init__()

        # Set seed
        if seed:
            self.seed = torch.manual_seed(seed)
        
        # Create input layer
        self.input_layer = nn.Linear(state_size, hidden[0])

        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for k in range(len(hidden)-1):
            self.hidden_layers.append(nn.Linear(hidden[k], hidden[k+1]))

        # Create output layers
        self.output_layer = nn.Linear(hidden[-1], action_size)

        return

    def forward(self, state):
        """
        Build a network that maps state -> action values
        
        Args:
            state (vector): Environment observation
        
        Returns:
            Model action values (Tensor)

        """

        # Feed forward input layer
        x = F.relu(self.input_layer(state))

        # Feed forward hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)

