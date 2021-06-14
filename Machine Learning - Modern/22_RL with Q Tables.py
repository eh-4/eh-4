# -*- coding: utf-8 -*-
""" Machine Learning Exercise 22 - Reinforcement Learning - Emerson Ham
This exercise covers the use of Q tables for reinforcement learning
"""

from time import sleep
from IPython.display import clear_output
import random

import gym
import numpy as np
np.random.seed(0)

# We will be using [OpenAI's gym](https://gym.openai.com/docs/) for rendering environments and we will specifically use the [Taxi-v2](https://gym.openai.com/envs/Taxi-v2/) environment for this exercise.
# Load the Taxi-v2 environment
env = gym.make("Taxi-v3").env

# Standardize expected results
env.seed(0)
env.reset()

print(f"Current State: {env.s}")
env.render()

print(f"The action space is discrete with {env.action_space.n} possibilities.")
print(f"The observation (state) space is discrete with {env.observation_space.n} possibilities.")

def initialize_q_table(env):
    # Initializes a Q table for an environment with all 0s
    # Inputs:
    #   env (gym.envs): The environment
    # Returns:
    #   np.array: The Q table

    qTable = np.zeros((env.observation_space.n, env.action_space.n))
    return qTable

def select_action(q_row, method, epsilon=0.5):
    # Selects the appropriate action given a Q table row for the state and a chosen method
    # Inputs:
    #   q_row (np.array): The row from the Q table to utilize
    #   method (str): The method to use, either "random" or "epsilon"
    #   epsilon (float, optional): Defaults to 0.5. The epsilon value to use for epislon-greed action selection
    # Raises:
    #   NameError: If method specified is not supported
    # Returns:
    #   int: The index of the action to apply

    if method not in ["random", "epsilon"]:
        raise NameError("Undefined method.")
    
    # YOUR CODE HERE
    action = 0;
    if method=="random":
        action = random.randint(0, len(q_row)-1)
    elif method=="epsilon":
        if random.random() < epsilon:
            #Explore
            action = random.randint(0, len(q_row)-1)
        else:
            #Exploit
            action = np.argmax(q_row)
    else:
        print("Something Weird Happened in select_action()")
    return action

# Print what happens if a step is taken with a given action
action = 0
vals = env.step(action)
print(f"An example returned from a step with action 0")
print(vals)
print(f"This returns the new state {vals[0]}, the reward received ({vals[1]}) based on performing the action {action}, whether or not the task has been completed, {vals[2]}, and some additional miscellaneous info.")

def calculate_new_q_val(q_table, state, action, reward, next_state, alpha, gamma):
    # Calculates the updated Q table value for a particular state and action given the necessary parameters
    # Inputs:
    #   q_table (np.array): The Q table
    #   state (int): The current state of the simulation's index in the Q table
    #   action (int): The current action's index in the Q table
    #   reward (float): The returned reward value from the environment
    #   next_state (int): The next state of the simulation's index in the Q table (Based on the environment)
    #   alpha (float): The learning rate
    #   gamma (float): The discount rate
    # Returns:
    #   float: The updated action-value expectation for the state and action

    qNew = (1-alpha)*q_table[state,action] + alpha*(reward + gamma*np.amax(q_table[next_state,:]))
    return qNew

# Set up hyperparameters
epsilon1_params = {
    "method": "epsilon",
    "epsilon": 0.1,
    "alpha": 0.1,
    "gamma": 0.5
}

epsilon2_params = {
    "method": "epsilon",
    "epsilon": 0.3,
    "alpha": 0.1,
    "gamma": 0.5
}

def train_sim(env, params, n=100):
    # Trains a simulation on an environment and return its Q table
    # Inputs
    #   env (gym.envs): The environment to train in
    #   params (dict): The parameters needed to train the simulation: method (for action selection), epsilon, alpha, gamma
    #   n (int, optional): Defaults to 100. The number of simulations to run for training
    # Returns:
    #   np.array: The trained Q table from the simulation

    my_q = initialize_q_table(env)
    
    for i in range(n):
        current_state = env.reset()
        done = False
        
        while not done:
            # Get the next action based on current state
            # Step through the environment with the selected action
            # Update the qtable
            action_num = select_action(my_q[current_state,:], params["method"],params["epsilon"])
            vals = env.step(action_num)
            next_state = vals[0]
            reward = vals[1]
            done = vals[2]
            my_q[current_state,action_num] = calculate_new_q_val(my_q,current_state,action_num,reward,next_state,params["alpha"], params["gamma"])

            # Prep for next iteration
            current_state = next_state 

        if (i+1) % 100 == 0:
            print(f"Simulation #{i+1:,} complete.")
        
    return my_q

# Train the two models for the two sets of hyperparameters
n = 10000
epsilon1_q = train_sim(env, epsilon1_params, n)
epsilon2_q = train_sim(env, epsilon2_params, n)

def test_sim(env, q_table, n=100, render=True):
    # Test an environment using a pre-trained Q table
    # Inputs:
    #   env (gym.envs): The environment to test
    #   q_table (np.array): The pretrained Q table
    #   n (int, optional): Defaults to 100. The number of test iterations to run
    #   render (bool, optional): Defaults to False. Whether to display a rendering of the environment
    # Returns:
    #   np.array: Array of length n with each value being the cumulative reward achieved in the simulation

    rewards = []
    
    for i in range(n):
        current_state = env.reset()

        tot_reward = 0
        done = False
        step = 0

        while not done:
            # Determine the best action
            # Step through the environment
            action = np.argmax(q_table[current_state,:])
            vals = env.step(action)
            next_state = vals[0]
            reward = vals[1]
            done = vals[2]

            tot_reward += reward
            step +=1
            if render:
                clear_output(wait=True)
                print(f"Simulation: {i + 1}")
                env.render()
                print(f"Step: {step}")
                print(f"Current State: {current_state}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Total rewards: {tot_reward}")
                sleep(.2)
            if step == 50:
                print("Agent got stuck. Quitting...")
                sleep(.5)
                break
            current_state = next_state
        
        rewards.append(tot_reward)
    
    return np.array(rewards)

# Test the models
# Add render=True to see the simulation running
epsilon1_rewards = test_sim(env, epsilon1_q, 10)
epsilon2_rewards = test_sim(env, epsilon2_q, 10)

print(f"The first epsilon greedy training method was able to get a median reward of {np.median(epsilon1_rewards)}.")
print(f"The second epsilon greedy training method was able to get a median reward of {np.median(epsilon2_rewards)}.")