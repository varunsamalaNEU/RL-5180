import numpy as np
from enum import IntEnum
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
    '''Simulate function for Four Rooms environment'''
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
    ]  # Walls as per starter code
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

    print(f"Possible actions from state {state} with action {action.name}:")
    print(f"  - {action.name}: 80%")
    for perp_action in action_outcomes[action]:
        print(f"  - {perp_action.name}: 10%")

    rand = np.random.rand() #if else and elif conditions for probabilities outlined in The EX-0
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

    reward = 1.0 if next_state == goal_state else 0.0 #Specifying reward for reaching 10,10 block

    print(f"Action taken: {action_taken.name}")
    print(f"Next State: {next_state}, Reward: {reward}")

    return next_state, reward

# REQ O/P
next_state, reward = simulate((0, 10), Action.UP)
print("Next State:", next_state, "Reward:", reward)

