from typing import List
import numpy as np
from enum import IntEnum
import random
import matplotlib.pyplot as plt

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: Action) -> tuple[int, int]:
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

def reset() -> tuple[int, int]:
    return (0, 0)

def random_policy(state: tuple[int, int]) -> Action:
    return Action(random.randint(0, 3))

def simulate(state: tuple[int, int], action: Action) -> tuple[tuple[int, int], float]:
    walls = [
        (0, 5), (2, 5), (3, 5), (4, 5),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
        (6, 4), (7, 4), (9, 4), (10, 4),
    ]
    goal_state = (10, 10)

    if state == goal_state:
        return reset(), 0

    # Introducing stochasticity in action outcomes
    action_outcomes = {
        Action.LEFT: [Action.LEFT],
        Action.DOWN: [Action.DOWN],
        Action.RIGHT: [Action.RIGHT],
        Action.UP: [Action.UP]
    }

    rand = np.random.rand() 
    if rand < 1:
        action_taken = action

    dx, dy = actions_to_dxdy(action_taken)
    next_state = (state[0] + dx, state[1] + dy)

    # Boundary wall evaluation
    if next_state in walls or not (0 <= next_state[0] <= 10 and 0 <= next_state[1] <= 10):
        next_state = state

    reward = 1.0 if next_state == goal_state else 0.0

    return next_state, reward

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

def main():
    #simulation with random policy
    steps_per_trial = 10000
    trials = 10
    all_trial_rewards = []

    for trial in range(trials):
        trial_rewards = run_simulation_with_random_policy(trial, steps_per_trial)
        all_trial_rewards.append(trial_rewards)

    # Plotting
    plt.figure(figsize=(12, 8))
    for rewards in all_trial_rewards:
        plt.plot(rewards, linestyle='--', alpha=0.5)  # Plot each trial with a dotted line

    # Calculate and plot the average cumulative reward at each step
    average_rewards = [sum(x) / trials for x in zip(*all_trial_rewards)]
    plt.plot(average_rewards, color='black', linewidth=2)  # Thick solid line for the average

    plt.title("Cumulative Reward over 10 Trials with Random Policy")
    plt.xlabel(f"Steps = {steps_per_trial}")
    plt.ylabel("Cumulative Reward")
    plt.show()

if __name__ == "__main__":
    
    main()
    

