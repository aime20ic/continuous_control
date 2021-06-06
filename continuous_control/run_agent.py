import numpy as np
from unityagents import UnityEnvironment



def parse_args():
    """
    Parse specified arguments from command line

    Args:
        None

    Returns:
        Argparse NameSpace object containing command line arguments

    """
    parser = argparse.ArgumentParser(description='Agent hyperparameters')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    args = parser.parse_args()

    return args

def main(args):
    """
    Run agent in environment

    Args:
        args: Argparse NameSpace object containing command line arguments

    Returns:
        None

    """

    # select this option to load version 1 (with a single agent) of the environment
    env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

    # select this option to load version 2 (with 20 agents) of the environment
    # env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()

        return


if __name__ == "__main__":
    """
    Execute script
    """
    args = parse_args()
    main(args)
