import numpy as np
import pandas as pd 
from stable_baselines3.common.callbacks import BaseCallback
import os 
import gymnasium as gym
from train_agents.util_functions import calculate_sharpe_ratio

class EvaluationCallback(BaseCallback):
    """ 
    This class implements different methods to evaluate the policy (e.g.
    calculating the average episodic return every k-th step). Furthermore,
    the best performing agent will be saved. 

    For this class the information in Stable-Baselines3 documentation is used:
    https://stable-baselines3.readthedocs.io/en/v1.0/guide/callbacks.html
    """

    def __init__(
                self, 
                eval_env:gym.Env, 
                eval_freq:int=10_000, 
                n_eval_episodes:int=10, 
                log_path:str=None, 
                verbose:int=1,
                save_every_nstep:bool=True,
                saving_freq:int=50_000,
                deterministic:bool=False,
                additional_name_eval_file:str=None,
                save_best_model:bool=True
                ):
        """ 
        Params:
            - eval_env (gym.Env): environment in which the agent will be 
            evalued
            - eval_freq (int): evaluate the agent every eval_freq steps
            - n_eval_episodes (int): number of episodes to run the policy 
            and evaluate it (the mean reward and standard deviation will
            be colleceted episodically)
            - log_path (str): where to store the best model and the policy
            metrics (mean reward, mean sharpe ratio and the corresponding
            standard deviations)
            - verbose (int): flag whether output should be shown (e.g.
            print statements)
            - save_every_nstep (bool): flag whether the model should be 
            stored every n-steps. If true, one needs to specify the number
            n
            - saving_freq (int): parameter n that is needed for saving
            the model every n-steps
            - deterministic (bool): flag whether one wants to use the 
            deterministic or stochastic policy. Default is set to False
            so that the stochastic policy is used
            - additional_name_eval_file (str): additional string used
            in the name of the evaluation csv file
            - save_best_model (bool): whether to save the on the 
            evaluation environment best performing model
        """
        # Define the user settings
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.save_every_nstep = save_every_nstep
        self.saving_freq = saving_freq
        self.deterministic = deterministic
        self.additional_name_eval_file = additional_name_eval_file
        self.save_best_model = save_best_model

        # Define metrics to show in the output dataframe
        self.evaluations_timesteps = []
        self.evaluations_rewards = []
        self.evaluations_std = []
        self.mean_sharpe_ratios = []
        self.std_sharpe_ratios = []

        # Set the best mean reward
        self.best_mean_reward = -np.inf

    def _evaluate_policy(self):
        """ 
        Run the policy for self.n_eval_episodes times and calculate the
        episodic mean of the rewards & sharpe ratio as well as the standard
        deviation of these metrics along the different episodes.

        Note: If the mean reward surpasses the previous mean reward, the
        agent will be stored since it achieves a higher return and is 
        therefore better.
        
        
        Furthermore, note that the funcationality on how to run the policy
        is taken from the 'evaluate_policy' method in SB3. (Code can be
        found here: https://github.com/DLR-RM/stable-baselines3/blob/master/\
            stable_baselines3/common/evaluation.py#L11) as well as the example
        to run the policy here: (Code: https://sb3-contrib.readthedocs.io/\
            en/master/modules/ppo_recurrent.html

        (Please note that since we also want to compute the sharpe-ratio,
        the method evaluate_policy from sb3 common.evaluation can
        not be used. Otherwise, the sharpe ratio and the average returns
        might refer to a different environment (due to resetting))

        Params:
            - None
        """
        # Create empty lists to store the episodic rewards and sharpe ratios
        episode_rewards = []
        sharpe_ratios = []
        
        # Run the policy n-times
        for _ in range(self.n_eval_episodes):

            # Reset the environment
            obs, _ = self.eval_env.reset()
            done, truncated, total_reward = False, False, 0
            rewards = [] # Save rewards for sharpe ratio
            states = None
            episode_starts = [True]

            # Run the policy until done or truncated
            while not done and not truncated:
                # Get the action and execute it
                action, states = self.model.predict(
                                        obs, 
                                        state=states,
                                        episode_start=episode_starts,
                                        deterministic=self.deterministic
                                        ) 
                new_obs, reward, done, truncated, _ = self.eval_env.step(
                                                                    action
                                                                    )

                episode_starts = [done]

                # Adjust the total reward per episode
                total_reward += reward
                rewards.append(reward)

                obs = new_obs
            
            # Calculate and save the sharpe ratio
            sharpe_ratio = calculate_sharpe_ratio(
                                    self.eval_env.unwrapped.asset_memory
                                    )
            sharpe_ratios.append(sharpe_ratio)

            # Save the reward per episode
            episode_rewards.append(total_reward)
        
        # Calculate the mean sharpe ratio and the standard deviation 
        mean_sharpe_ratio = np.mean(sharpe_ratios)
        std_sharpe_ratio = np.std(sharpe_ratios)

        # Calculate the mean and the standard deviation of the episodic return
        mean, std = np.mean(episode_rewards), np.std(episode_rewards)

        # Check if the mean episodic return is higher than before
        if mean > self.best_mean_reward and self.save_best_model:
            # Save the model  
            self.model.save(os.path.join(self.log_path, "best_model"))
            
            # Print that a new best model was found (if verbose activated)
            if self.verbose >= 1:
                print(f'New best model with mean reward: {round(mean)}')
            
            # Overwrite the best mean model
            self.best_mean_reward = mean

        # Save the timestep, mean and standard deviation
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_rewards.append(mean)
        self.evaluations_std.append(std)
        self.mean_sharpe_ratios.append(mean_sharpe_ratio)
        self.std_sharpe_ratios.append(std_sharpe_ratio)
    
    def _on_step(self) -> bool:
        """ 
        On each k-th step, the current policy will be evaluated. 
        
        Note: Since the initial environment could be a vectorized environment 
        and therefore only the timesteps in the series [n_vec, 2*n_vec, ...] 
        will be encountered, one needs to make sure that self.eval_freq is part 
        of the series. Otherwise, the policy will never be evaluated. The same 
        holds true for saving the best model.
        """

        # Evaluate the policy every k-th step
        if self.num_timesteps % self.eval_freq == 0:                
            self._evaluate_policy()

        # Check if the model needs to be saved
        if self.save_every_nstep and self.num_timesteps % self.saving_freq == 0:
            self.model.save(os.path.join(self.log_path,
                                    f'models/model_step_{self.num_timesteps}'))

        # Return True, otherwise the training will be aborted
        return True

    def _on_training_start(self) -> None:
        """
        Evaluate the policy to have a performance value at timestep=0 
        and create folders to store the 
        
        Params:
            - None
        """
        # Create subfolder to store the best model
        os.makedirs(self.log_path, exist_ok=True)

        # If the models should be stored episodically, a new folder is created
        # in which the agents will be stored
        if self.save_every_nstep:
            # Create folder
            os.makedirs(self.log_path + '/models', exist_ok=True)

            # Save the first model (to have the agent at timestep t=0)
            self.model.save(os.path.join(self.log_path,
                            f'models/model_step_{self.num_timesteps}'))

        # Evaluate the policy
        self._evaluate_policy()

    def _on_training_end(self) -> None:
        """ 
        Once the training is done, the simulated metrics will be stored as 
        a csv-file.

        Params:
            - None
        Returns:
            - None (instead, a file will be stored)
        """
        # Define a pandas dataframe based on the defined metrics
        df = pd.DataFrame({
                         'timesteps':self.evaluations_timesteps,
                         'rewards_mean':self.evaluations_rewards,
                         'rewards_std':self.evaluations_std,
                         'sharpe_ratio_mean':self.mean_sharpe_ratios,
                         'sharpe_ratio_std':self.std_sharpe_ratios
                         })
        
        # Save the csv file in the specified location
        df.to_csv(self.log_path + 
                  f'/eval_{self.additional_name_eval_file}.csv')