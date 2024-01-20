import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import random

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


def simulate(state: Tuple[int, int], action: Action):
    
    # Walls are listed for you
    # Coordinate system is (x, y) 
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
        (-1,-1),
        (-1,0),
        (-1,1),
        (-1,2),
        (-1,3),
        (-1,4),
        (-1,5),
        (-1,6),
        (-1,7),
        (-1,8),
        (-1,9),
        (-1,10),
        (-1,11),
        (0,11),
        (1,11),
        (2,11),
        (3,11),
        (4,11),
        (5,11),
        (6,11),
        (7,11),
        (8,11),
        (9,11),
        (10,11),
        (11,11),
        (11,10),
        (11,9),
        (11,8),
        (11,7),
        (11,6),
        (11,5),
        (11,4),
        (11,3),
        (11,2),
        (11,1),
        (11,0),
        (11,-1),
        (10,-1),
        (9,-1),
        (8,-1),
        (7,-1),
        (6,-1),
        (5,-1),
        (4,-1),
        (3,-1),
        (2,-1),
        (1,-1),
        (0,-1),

    ]

    goal_state = (10, 10)


    if action == Action.UP or Action.DOWN:
        actions = [action, Action.LEFT, Action.RIGHT]
        probability = [0.9, 0.05, 0.05]
        noisy_action = np.random.choice(actions, p=probability)
    if action == Action.LEFT or Action.RIGHT:
        actions = [action, Action.UP, Action.DOWN]
        probability = [0.9, 0.05, 0.05]
        noisy_action = np.random.choice(actions, p=probability)


    final_action = actions_to_dxdy(noisy_action)

    if state == goal_state:
        next_state = reset()
    else:
        next_state = (state[0] + final_action[0],state[1] + final_action[1])
        if next_state in walls:
            next_state = state
 
    reward = 0
    if next_state == goal_state:
        reward = 1
        

    return next_state, reward



def agent(
    steps: int = 10000,
    trials: int = 10,
    policy=Callable[[Tuple[int, int]], Action],
):
  
    
    rewards = []

    for t in range(trials):
        state = reset()
        i = 0

        total_rewards = []
        total_reward = 0
        while i < steps:

           
            action = policy(state)

            
            next_state, reward = simulate(action=action, state=state)
            state = next_state

            total_reward = total_reward + reward
            total_rewards.append(total_reward)

            i = i + 1

            # Print information for each step
            print(f"Trial {t + 1}, Step {i}:")
            print(f"  Action: {action}")
            print(f"  Next State: {state}")
            print(f"  Reward: {reward}")
            print(f"  Cumulative Reward: {total_reward}")
            print("=" * 30)

        rewards.append(total_rewards)

    

    return rewards
#Q3

def random_policy(state: tuple[int, int]) -> Action:
    return Action(random.randint(0, 3))
def run_simulation_with_random_policy(trial, steps):
    state = reset()  
    cumulative_reward = 0  
    rewards_per_step = []

    for step in range(steps):
        action = random_policy(state)  
        state, reward = simulate(state, action)  
        cumulative_reward += reward
        rewards_per_step.append(cumulative_reward)

        if state == (10, 10):  
            state = reset()  
        
        print(f"Trial {trial + 1}, Step {step + 1}: Selected Action - {action.name}, Reward - {reward}, Cumulative Reward - {cumulative_reward}")

    return rewards_per_step

# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    return Action.RIGHT


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
        (-1,-1),
        (-1,0),
        (-1,1),
        (-1,2),
        (-1,3),
        (-1,4),
        (-1,5),
        (-1,6),
        (-1,7),
        (-1,8),
        (-1,9),
        (-1,10),
        (-1,11),
        (0,11),
        (1,11),
        (2,11),
        (3,11),
        (4,11),
        (5,11),
        (6,11),
        (7,11),
        (8,11),
        (9,11),
        (10,11),
        (11,11),
        (11,10),
        (11,9),
        (11,8),
        (11,7),
        (11,6),
        (11,5),
        (11,4),
        (11,3),
        (11,2),
        (11,1),
        (11,0),
        (11,-1),
        (10,-1),
        (9,-1),
        (8,-1),
        (7,-1),
        (6,-1),
        (5,-1),
        (4,-1),
        (3,-1),
        (2,-1),
        (1,-1),
        (0,-1),

    ]

    # mapping for the wall detections around the agent and taking actions accordingly
    action_map = {

        #up,down,left,right
        (0, 0, 0, 0)    : np.random.choice([Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]),
        (1, 0, 0, 0)    : np.random.choice([Action.LEFT, Action.RIGHT]),
        (0, 1, 0, 0)    : np.random.choice([Action.LEFT, Action.RIGHT]),
        (0, 0, 1, 0)    : np.random.choice([Action.UP, Action.DOWN]),
        (0, 0, 0, 1)    : np.random.choice([Action.UP, Action.DOWN]),
        (1, 0, 1, 0)    : np.random.choice([Action.DOWN, Action.RIGHT]),
        (1, 0, 0, 1)    : np.random.choice([Action.DOWN, Action.LEFT]),
        (0, 1, 1, 0)    : np.random.choice([Action.UP, Action.RIGHT]),
        (0, 1, 0, 1)    : np.random.choice([Action.UP, Action.LEFT]),
        (1, 1, 0, 0)    : np.random.choice([Action.RIGHT, Action.LEFT]),
        (0, 0, 1, 1)    : np.random.choice([Action.UP, Action.DOWN]),

    }

    # checking for the walls around the agent and taking action
    action = action_map[(state[0], state[1]+1) in walls, (state[0], state[1]-1) in walls, (state[0]-1, state[1]) in walls, (state[0]+1, state[1]) in walls]

    return action

    


def main():
    

 # simulation with random policy
    steps_per_trial = 10000
    trials = 10
    all_trial_rewards = []

    for trial in range(trials):
        trial_rewards = run_simulation_with_random_policy(trial, steps_per_trial)
        all_trial_rewards.append(trial_rewards)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot random policy trials
    for rewards in all_trial_rewards:
        plt.plot(rewards, linestyle='--', alpha=0.5)  # Plot each trial with a dotted line

    # Calculate and plot the average cumulative reward at each step
    average_rewards = [sum(x) / trials for x in zip(*all_trial_rewards)]
    plt.plot(average_rewards, color='black', linewidth=2, label='Random Policy')  # Thick solid line for the average

    # worse policy
    worst_rewards = agent(policy=worse_policy)
    avg_worst_rewards = np.mean(worst_rewards, axis=0)

    # plots for worse policy
    for r in worst_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000), avg_worst_rewards, linewidth='3', color='green', label='Worse Policy')

    # better policy
    better_rewards = agent(policy=better_policy)
    avg_better_rewards = np.mean(better_rewards, axis=0)

    # plots for better policy
    for r in better_rewards:
        plt.plot(range(10000), r, ':')
    plt.plot(range(10000), avg_better_rewards, linewidth='3', color='blue', label='Better Policy')

    plt.title("Cumulative Reward over 10 Trials")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()