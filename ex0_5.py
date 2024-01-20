from typing import List, Tuple
import numpy as np
from enum import IntEnum
import random
import matplotlib.pyplot as plt

class Action(IntEnum):
    # Enum for directions
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

goal_state = None

def set_random_goal(walls):
    # Function to set a random goal excluding walls and start position
    global goal_state
    possible_goals = [(x, y) for x in range(11) for y in range(11) if (x, y) not in walls and (x, y) != (0, 0)]
    goal_state = random.choice(possible_goals)

def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    # Map actions to movements
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]

def reset() -> Tuple[int, int]:
    # Reset function
    return (0, 0)

def random_policy(state: Tuple[int, int]) -> Action:
    # Random policy choosing from the four directions
    return Action(random.randint(0, 3))

def simulate(state: Tuple[int, int], action: Action) -> Tuple[Tuple[int, int], float]:
    # Simulate function for the Four Rooms environment
    walls = [
        (0, 5), (2, 5), (3, 5), (4, 5),
        (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
        (6, 4), (7, 4), (9, 4), (10, 4),
    ]
    global goal_state

    if state == goal_state:
        return reset(), 1

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

    if next_state in walls or not (0 <= next_state[0] <= 10 and 0 <= next_state[1] <= 10):
        next_state = state

    reward = 1.0 if next_state == goal_state else 0.0

    if not isinstance(action_taken, Action):
        action_taken = Action(action_taken)

    print(f"Action taken: {action_taken.name}")
    print(f"Next State: {next_state}, Reward: {reward}")

    return next_state, reward

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.25, epsilon=0.4):
        # Q-learning agent class
        self.q_table = np.zeros((11, 11, len(Action)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        # Choose action based on epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(Action))
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Q-learning update rule
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

def qlearning_poli(steps: int, trials: int, walls: List[Tuple[int, int]]) -> List[List[float]]:
    # Run Q-learning trials
    all_trials_rewards = []
    agent = QLearningAgent()

    for _ in range(trials):
        state = reset()
        cumulative_reward = 0
        trial_rewards = []
        set_random_goal(walls)

        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward = simulate(state, action)
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            cumulative_reward += reward
            trial_rewards.append(cumulative_reward)

            if state == goal_state:
                state = reset()

        all_trials_rewards.append(trial_rewards)

    return all_trials_rewards


def run_trials(agent, trials, steps, walls):
    for _ in range(trials):
        set_random_goal(walls)
        state = reset()
        first_goal_reached = False
        steps_to_first_goal = 0

        for step in range(steps):
            action = agent.choose_action(state)
            next_state, reward = simulate(state, action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            if not first_goal_reached and state == goal_state:
                first_goal_reached = True
                steps_to_first_goal = step

# Testing!
trials = 10
agent = QLearningAgent()

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
    ] 
run_trials(agent, 10, 10000, walls)
qlearning_rewards = qlearning_poli(10000, 10, walls)

# Plotting
plt.figure(figsize=(12, 8))

# Plot each trial with a dotted line
for i, rewards in enumerate(qlearning_rewards):
    plt.plot(rewards, linestyle='--', alpha=0.5, label=f'Trial {i + 1}')
    print(f"Cumulative Rewards for Trial {i + 1}: {rewards[-1]}")  # Print the final cumulative reward for each trial

# Calculate and plot the average cumulative reward at each step
average_rewards = [sum(x) / trials for x in zip(*qlearning_rewards)]
plt.plot(average_rewards, color='black', linewidth=2, label='Average Reward')  # Thick solid line for the average
print(f"Average Cumulative Rewards: {average_rewards[-1]}")  # Print the final average cumulative reward

plt.title("Cumulative Reward over 10 Trials with Q-Learning Policy")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()


