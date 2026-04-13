### This file implements different functions that are needed ###
### for the evaluation and training of the agent             ###

import numpy as np
import pandas as pd
import gymnasium as gym
import optuna
from torch import nn as nn

def evaluate_policy(
                    eval_env:gym.Env, 
                    model, 
                    n_eval_episodes:int= 5,
                    deterministic:bool=False
                    ):
    """ 
    Run the policy for n episodes, calculate the mean episodic reward and 
    sharpe ratio. Furthermore, compute the standard deviation of these
    metrics along the different episodes.
     
    Furthermore, note that the funcationality on how to run the policy
    is taken from the 'evaluate_policy' method in SB3. (Code can be
    found here: https://github.com/DLR-RM/stable-baselines3/blob/master/\
        stable_baselines3/common/evaluation.py#L11) as well as the example
    to run the policy here: (Code: https://sb3-contrib.readthedocs.io/\
            en/master/modules/ppo_recurrent.html

    Params:
        - eval_env (gym.Env): environment in which the agent should be
        evaluated
        - model: policy of the agent which was trained with SB3
        - n_eval_episodes (int): number of episodes to run the policy
        - deterministic (bool): whether to use the stochastic or
        deterministic policy. If set to False, the stochastic policy
        is being used
    """

    # Create empty lists to store important metrics
    episode_rewards = []
    sharpe_ratios = []
    asset_memories = []
    
    # Run the policy specified number of times
    for _ in range(n_eval_episodes):

        # Reset the environment
        obs, _ = eval_env.reset()
        done, truncated, total_reward = False, False, 0
        rewards = [] # Save rewards for sharpe ratio
        states = None
        episode_starts = [True]

        # Run the policy until done or truncated
        while not done and not truncated:
            # Get the action and execute it
            action, states = model.predict(
                                    obs, 
                                    state=states,
                                    episode_start=episode_starts,
                                    deterministic=deterministic
                                    ) 
            new_obs, reward, done, truncated, _ = eval_env.step(
                                                                action
                                                                )
            episode_starts = [done]

            # Adjust the total reward per episode
            total_reward += reward
            rewards.append(reward)

            obs = new_obs

        # Get the portfolio values over the time 
        asset_memory = eval_env.unwrapped.asset_memory

        # Calculate the sharpe ratio and save it
        sharpe_ratio = calculate_sharpe_ratio(eval_env.unwrapped.asset_memory)
        sharpe_ratios.append(sharpe_ratio)

        # Save the reward per episode
        episode_rewards.append(total_reward)

        # Save the portfolio values over the time
        asset_memories.append(asset_memory)

    # Calculate the mean sharpe ratio and the standard deviation 
    mean_sharpe_ratio = np.mean(sharpe_ratios)
    std_sharpe_ratio = np.std(sharpe_ratios,ddof=1)

    # Calculate the mean and the standard deviation of the episodic return
    mean, std = np.mean(episode_rewards), np.std(episode_rewards)

    return asset_memories, mean_sharpe_ratio, std_sharpe_ratio, mean, std

def run_policy(
                eval_env:gym.Env, 
                model, 
                n_eval_episodes:int= 5,
                deterministic:bool=False
                ):
    """ 
    This method is the same as 'evaluate_policy' above but instead
    of returning the portfolio values over the time, the mean sharpe
    ratio and mean episodic reward (as well as the standard) deviation,
    this function returns the number of shares (for each stock) over
    time as well as the prices of the stocks and the cash. 

    (For further notes regarding the implementation, see the method
    description on the upper method. Furthermore, note that this 
    method is needed to analyse the trading strategy of the agent.)

    Params:
        - eval_env (gym.Env): environment in which the agent should be
        evaluated
        - model: policy of the agent which was trained with SB3
        - n_eval_episodes (int): number of episodes to run the policy
        - deterministic (bool): whether to use the stochastic or
        deterministic policy. If set to False, the stochastic policy
        is being used
    """
    # Create empty lists to store relevant information for each timestep
    # the number of shares, prices and the cash position will be stored
    shares = []
    prices = []
    cash = []

    # Run the policy n-times
    for num_iteration in range(n_eval_episodes):
        # Reset the environment
        obs, _ = eval_env.reset()
        done, truncated = False, False
        states = None
        episode_starts = [True]

        # Define empty list to store the number of shares & cash within
        # the iteration
        shares_iteration = []
        cash_iteration = []

        # Run the policy until done or truncated
        while not done and not truncated:
            # Get the action and execute it
            action, states = model.predict(
                                    obs, 
                                    state=states,
                                    episode_start=episode_starts,
                                    deterministic=deterministic
                                    ) 
            new_obs, reward, done, truncated, info = eval_env.step(
                                                                action
                                                                )
            
            # Save the number of shares, cash and prices
            shares_iteration.append(info['shares'])
            cash_iteration.append(info['cash'])
            if num_iteration == 0:
                prices.append(info['prices'])

            episode_starts = [done]
            obs = new_obs
        
        # Store the shares and cash per iteration in the overall list
        shares.append(shares_iteration)
        cash.append(cash_iteration)

    return shares,prices,cash

def calculate_sharpe_ratio(asset_memory:list):
    """
    Calculate the annualized Sharpe Ratio for a given list of total 
    assets. Note that it is assumed that the calculated returns
    are daily returns.
    
    Params:
        - asset_memory (list): containing the total assets
    Returns:
        - sharpe ratio (float): annualized sharpe ratio
    """
    # Convert the asset memory to a pandas dataframe
    market_data_total_value = pd.DataFrame(asset_memory)
    
    # Calculate the percentage change (return)
    pct_change = market_data_total_value.pct_change()

    # Calculate the annualized sharpe ratio based on the daily returns 
    return (252**0.5)*pct_change.mean()/pct_change.std()

def calculate_cumulative_returns(asset_memories:list):
    """ 
    Calculate the mean cumulative return and standard deviation over 
    n-runs of the policy. 

    Params:
        - asset_memories (list): list of sublists, where each sublist 
        contains the portfolio value for one specific run. The shape of 
        the list is therefore (number runs, length episode)
    Return:
        - mean (float) of the cumulative return at each timestep
        - standard deviation (float) of the cumulative return at each timestep
    """
    # Convert list to numpy array for easier indexing & calculation
    asset_memories=np.array(asset_memories)

    # Calculate cumulative returns
    initial_values = asset_memories[:, 0].reshape(-1, 1)
    cumulative_returns = (asset_memories - initial_values) / initial_values

    # Calculate the mean and the standard deviation over the simulated cumulative returns
    return cumulative_returns.mean(axis=0), cumulative_returns.std(axis=0)

def sample_recurrent_ppo_params(trial: optuna.Trial):
    """
    This method samples the hyperparameters for the RecurrentPPO model. Note
    that this method is based on the FinRL repository, where an example is 
    given on how to sample the parameters using Optuna 
    (finrl/agents/stablebaselines3/hyperparams_opt.py). Because the example is 
    just refers to the PPO algorithm and not RecurrentPPO, the method was expanded 
    to sample further algorithm specific parameters like the hidden size of the
    LSTM network.

    Params:
        - trial (optuna.Trial)
    Returns:
        - Paramaeters for the RecurrentPPO Algorithm (e.g. batch size, n_steps,
        gamma,...)
    """
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    n_steps = trial.suggest_categorical(
        "n_steps", [128, 256]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.99, 0.999]
    )

    learning_rate = trial.suggest_categorical("learning_rate", [1e-4,3e-4,5e-4])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.95, 0.98, 0.99]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.4, 0.5]
    )
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size",[64,128,256])

    net_arch = {
        "small":dict(pi=[64, 64], vf=[64, 64]), # default configuration
        "medium":dict(pi=[256, 64], vf=[256, 64]),
    }[net_arch]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": dict(
            net_arch=net_arch,
            lstm_hidden_size=lstm_hidden_size,
        ),
    }