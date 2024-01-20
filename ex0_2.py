import numpy as np
from enum import IntEnum
from typing import Callable 
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt

class Action(IntEnum):
    #Enum from starter code
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: Action) -> tuple[int, int]:
    # Defining movement across the board
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

def reset() -> tuple[int, int]:
    # Defining a Generic reset function
    return (0, 0)

def simulate(state: tuple[int, int], action: Action) -> tuple[tuple[int, int], float]:
    """Simulate function for Four Rooms environment"""
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
    ]  #  walls given in starter code
    goal_state = (10, 10)

    if state == goal_state:
        return reset(), 0

    # Coding the noise and its possible stochasity
    action_outcomes = {
        Action.LEFT: [Action.DOWN, Action.UP],
        Action.DOWN: [Action.LEFT, Action.RIGHT],
        Action.RIGHT: [Action.DOWN, Action.UP],
        Action.UP: [Action.LEFT, Action.RIGHT]
    }

    rand = np.random.rand() #if, else and elif conditions for probabilities outlined in ex0
    if rand < 0.8:
        action_taken = action
    elif rand < 0.9:
        action_taken = np.random.choice(action_outcomes[action])
    else:
        action_taken = np.random.choice(action_outcomes[action])

    dx, dy = actions_to_dxdy(action_taken)
    next_state = (state[0] + dx, state[1] + dy)

    # Boundary, wall Evaluation
    if next_state in walls or not (0 <= next_state[0] <= 10 and 0 <= next_state[1] <= 10):
        next_state = state

    reward = 1.0 if next_state == goal_state else 0.0

    return next_state, reward


def manual_policy(state: tuple[int, int]) -> Action:
    print(f"Current State: {state}")
    action = input("Choose an action (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP): ")
    try:
        action = Action(int(action))
        if action in Action:
            return action
        else:
            print("Invalid action. Please enter a valid action.")
            return manual_policy(state)
    except ValueError:
        print("Invalid input. Please enter a number corresponding to an action.")
        return manual_policy(state)
def agent(
    steps: int = 10000,
    trials: int = 1,
    policy: Callable[[tuple[int, int]], Action] = manual_policy,
):
    for t in range(trials):
        state = reset()
        cumulative_reward = 0
        i = 0

        while i < steps:
            action = policy(state)  # Select action to take based on the policy
            next_state, reward = simulate(state, action)  # Take a step in the environment
            cumulative_reward += reward  # Record the reward

            # Update state and step counter
            state = next_state
            i += 1

            if state == (10, 10):  # Check if the goal state is reached
                break

        print(f"Trial {t + 1}: Cumulative Reward = {cumulative_reward}")




def main():
    # with manual policy
    agent(steps=30, trials=1, policy=manual_policy)


if __name__ == "__main__":
    main()

