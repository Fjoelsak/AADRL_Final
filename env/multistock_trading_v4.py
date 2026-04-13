import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiStockTrading(gym.Env):
    """
    This class implements a trading environment designed as a gymnasium environment 
    in which an agent can trade multiple stocks simultaneously.
    
    Please note that this class is based on the FinRL repository, which contains the 
    code to the paper "Deep reinforcement learning for automated stock 
    trading: an ensemble strategy". The code can be found here:
    https://github.com/AI4Finance-Foundation/FinRL-Trading/commit/20b589847   

    Furthermore, this environment is very similar to the one presented in "Deep 
    Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy". One 
    major difference is that in the article, the agent sold all the shares once the 
    turbulence index exceeded a specified threshold. In this environment, the agent 
    will not follow this rule. Instead, the turbulence index is added to the state 
    space, allowing the agent to learn the relationships on its own without major 
    constraints. In this way, the agent will implicitly develop a strategy based on 
    the turbulence index.

    Main Assumptions of this environment:
        • The orders can be executed directly for the given close prices (full liquidity)
        • The agent has no impact on the stock market
        • The cash position can not be negative 
        • Transaction costs are incurred for each trade. These depend on the value
        of the transaction. For this, the agent will have to pay 0.1% of the whole
        transaction value.
        • Short selling is not allowed
        • Only a whole number shares can be purchased
    """

    # Define maximum amount of shares that can be traded (each day)
    HMAX_NORMALIZE = 100 

    # Define the initla account balance and the number of shares to trade
    INITIAL_ACCOUNT_BALANCE = 1000000 
    STOCK_DIM = 30 

    # Define the transaction fee in percent (0.1%)
    TRANSACTION_FEE_PERCENT = 0.001 

    # Define the reward scaling (only used when reward function is not the 
    # risk adjusted one)
    REWARD_SCALING = 1e-4 

    # Define the render modes
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
            self, 
            market_data:pd.DataFrame,
            turbulence_index:pd.DataFrame,
            consider_sentiments=False,
            consider_indicators=True,
            n_step_sharpe:int=None,
            render_mode=None,
            seed:int=None
            ):
        """
        Initialization of the environment

        Params:
            - market_data (pd.DataFrame): dataframe containing the financial time series
              data and the sentiments. The index should contain the corresponding date. 
              Please note that this dataframe contains a multi-index as a column (e.g.
              [Adj Close, AAPL; Sentiments, NKE; ...])
            - turbulence_index (pd.DataFrame): dataframe containing for each timestep
              of the market data, the turbulence index at that time. 
            - turbulence_threshold (float): Threshold at which the agent stops trading 
              when the turbulence index surpasses this value. This threshold controls the 
              agent's risk aversion.
            - consider_sentiments (bool): flag whether the sentiments should be considered
              or not into the state. If not, the columns containing the sentiments will
              be removed from the state. (Default: False)
            - consider_indicators (bool): flag whether the indicators (RSI, CCI, ADX) should
              be considered or not into the state. If not, the columns containing the
              indicators will be removed from the state. (Default: True)
            - n_step_sharpe (int): time window to use to calculate the sharpe ratio and 
              add it to the reward. If the agent has done less steps than specified, the
              reward is simply the absolute difference in total assets. Otherwise, the 
              sharpe ratio will be added or subtracted from the total portfolio value 
              differences. If the sharpe ratio should not be considered, the value should
              be set as None, which is the default value. (See the article: 'A Sharpe Ratio 
              Based Reward Scheme in Deep Reinforcement Learning for Financial Trading' by
              G. Rodinos, P. Nousi, N. Passalis and . A. Tefas)
        """
        # Get the data and the starting trading day
        self.market_data = market_data
        self.day = 0

        # Remove sentiments from dataset, if they should not be considered 
        if not consider_sentiments:
            filter = self.market_data.columns[
                self.market_data.columns.get_level_values(0) != 'sentiment'
                ]
            self.market_data = self.market_data[filter]

        if len(consider_indicators) != 0:
            filter = self.market_data.columns[
                ~self.market_data.columns.get_level_values(0).isin(consider_indicators)
                ]
            self.market_data = self.market_data[filter]
        
        # Set the turbulence index
        self.turbulence_index = turbulence_index

        # Derive number of stocks from data (supports universes smaller than 30)
        self.STOCK_DIM = len(self.market_data.columns.get_level_values(1).unique())

        # Definition the action space (n Stocks, and for stock a value between [-1,1])
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.STOCK_DIM,),seed=seed)


        # Get unique columns (Adj Close, MACD, ...)
        num_unique_cols  = len(
            self.market_data.columns.get_level_values(0).unique()
            )

        num_features = num_unique_cols
        
        # Definition of observation space (add 1 to the general num_features, since the 
        # number of shares per stock will be considered. Add on top of that 2, since the 
        # account balance and the turbulence index will be considered)
        self.observation_space = spaces.Box(
                                        low=-np.inf, 
                                        high=np.inf, 
                                        shape = ((num_features + 1) *self.STOCK_DIM + 2,)
                                        )
        
        # Set render mode 
        self.render_mode = render_mode

        # Remove the dates as the index
        self.market_data = self.market_data.reset_index(drop = True)

        # Get for starting day the market data (all rows where index is self.day)
        self.data = self.market_data.loc[self.day,:]

        # Initialize state (account balance + number shares + close prices + technical 
        # indicators + sentiments (if defined by the user)) 
        state_tmp = [self.INITIAL_ACCOUNT_BALANCE if i == 0 else 0 
                     for i in range(self.STOCK_DIM+1)]
        self.state = np.concatenate(
                                    [state_tmp, 
                                     self.data, 
                                     [self.turbulence_index.iloc[self.day,0]]
                                     ],dtype=np.float32)

        # Define the terminal variable 
        self.terminal = False             

        # Memorization of account balance & rewards in different list
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []

        # Define variable to set the previous sharpe ratio (needed for reward function)
        self.sharpe_ratio = -np.inf

        # Save the time window to calculate the reward
        self.n_step_sharpe = n_step_sharpe

    def step(self, actions):
        """ 
        Take a step within the environment.

        Params:
            - actions (np.array): this array contains for each stock the normalized 
            amount (within [-1,1]) of shares to buy or sell.
        Returns:
            - next state (list): list containing the next state after
              taking the action specified in actions 
            - reward (float): reward that the agent receives after taking
              the action and landing in the next state
            - done (bool): information whether a terminal state was entered
            - truncated (bool): information whether more steps than allowed
              were performed (This will be False in this environment but
              it might be needed in general when using SB3)
            - additional information (str): additional information about 
              the environment.
        """
        # Rescale the action (from [-1,1] to [-HMAX, HMAX])
        actions = actions * self.HMAX_NORMALIZE

        # Get the account balance and the portfolio value 
        account_balance = self.state[0]
        portfolio_value = np.dot(
            self.state[1:(self.STOCK_DIM+1)], # shares
            self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)] # prices
        )

        # Calculate the total asset value 
        begin_total_asset = account_balance + portfolio_value

        # Get the stocks that need to be bought and sold
        buy_indices, sell_indices = self._get_buy_sell_stocks(actions)
        
        for index in sell_indices:
            self._sell_stock(index, actions[index])
        for index in buy_indices:
            self._buy_stock(index, actions[index])
        
        # Adjust the day and get the corresponding data for the day
        self.day += 1
        self.data = self.market_data.loc[self.day,:]         

        # Set the new state      
        state_tmp = [self.state[0] if i == 0 else self.state[i]
                     for i in range(self.STOCK_DIM+1)]
        self.state = np.concatenate(
                                    [state_tmp, 
                                     self.data,
                                     [self.turbulence_index.iloc[self.day,0]]], 
                                     dtype=np.float32)

        # Get the account balance and the portfolio value after the trade 
        account_balance_end = self.state[0]
        portfolio_value_end = np.dot(
            self.state[1:(self.STOCK_DIM+1)], # shares
            self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)] # prices
        )

        # Get the assets after the trade
        end_total_asset = account_balance_end + portfolio_value_end

        # Save the total assets 
        self.asset_memory.append(end_total_asset)

        # Calculate the reward
        self.reward = self._calculate_reward(
                                        end_total_asset, 
                                        begin_total_asset,
                                        self.n_step_sharpe
                                        )

        self.rewards_memory.append(self.reward)
        
        # Check whether a terminal state is entered
        self.terminal = self.day >= len(self.market_data)-1

        # Get the info and transform the state to a numpy array
        info = self._get_info()
        self.state = np.array(self.state, dtype=np.float32)
        return self.state, self.reward, self.terminal, False, info

    def _sell_stock(self, index, action):
        """ 
        This method will sell as many stocks as specified in action for a given stock. 
        
        Please note that one can sell as many stocks as one has. 
        Example: If the action is -100 and the agent has only 80 stocks, 
        the total amount of stocks will be sold.

        Params:
            - index (int): index to identify the share (from 0 to self.STOCK_DIM-1)
            - action (float): number of shares to sell for the given stock. Please note
            that this variable is of type float, since the action is first chosen in 
            the interval [-1,1] and then rescaled to [-H_MAX, H_MAX]. Since we defined
            that there can only be the actions {-H_MAX, -H_MAX-1, ..., -1,0, 1,..., 
            H_MAX-1, H_MAX}, the action will be rounded to an integer. 
        """
        # Check if there are any shares that can be sold
        num_shares = self.state[index+1]
        if num_shares > 0:
            price = self.state[self.STOCK_DIM + index + 1]

            # Number of shares to sell (clip since short selling not allowed)
            shares_to_sell = min(round(abs(action)),num_shares)

            # Update the account balance, once the stock got sold
            self.state[0] += price * shares_to_sell * (1 - self.TRANSACTION_FEE_PERCENT)

            # Update the number of shares
            self.state[index+1] -= shares_to_sell
        else:
            pass

    def _buy_stock(self, index, action):
        """
        This method will buy the number of shares specified by the action for 
        the given stock. 
        
        Please note that one can only buy as many shares as one's available money 
        allows.

        Params:
            - index (int): index to identify the share (from 0 to self.STOCK_DIM-1)
            - action (float): number of shares to buy for the given stock. Please note
            that this variable is of type float, since the action is first chosen in 
            the interval [-1,1] and then rescaled to [-H_MAX, H_MAX]. Since we defined
            that there can only be the actions {-H_MAX, -H_MAX-1, ..., -1,0, 1,..., 
            H_MAX-1, H_MAX}, the action will be rounded to an integer.         
        """ 
        # Count how many shares could be bought (account in transaction cost. Otherwise,
        # the account balance could get negative after the trade)
        possible_shares = self.state[0] // (self.state[self.STOCK_DIM + index + 1] * \
                                            (1+ self.TRANSACTION_FEE_PERCENT))

        # Check that at least 1 share could be bought
        if possible_shares > 0:
            # Calculate the number of shares to buy
            shares_to_buy = min(possible_shares, round(action))
            
            # Update the account balance
            price = self.state[self.STOCK_DIM + index+1]
            self.state[0] -= price*shares_to_buy * (1+ self.TRANSACTION_FEE_PERCENT)

            # Update the amount of shares
            self.state[index+1] += shares_to_buy
        else:
            pass


    def _calculate_reward(
                          self,
                          end_total_assets:float,
                          begin_total_assets:float,
                          n_window:int=None
                          ):
        """ 
        This method calculates the reward that the agent receives after
        taking an action. There are 2 ways how the reward can get calculated:
        
            1) The reward is the absolute difference between the total assets at 
            timesteps t+1 and t. Please note that the transaction fees are also
            considered in this way and do not need to be subtracted since the 
            balance value at timestep t+1 is already being decreased by the fees. 
            This corresponds to the reward function in the article:
            "Deep reinforcement learning for automated stock trading: an ensemble 
            strategy"

            2) If the agent has already done n-steps within the environment, 
            we will calculate the sharpe ratio over the last n-steps and add the
            sharpe ratio on the absolute portfolio value change, if it increased.
            If the portfolio value decreased, we will subtract the sharpe ratio
            from the portfolio value. This idea is based on the article:
            "A Sharpe Ratio Based Reward Scheme in Deep Reinforcement Learning for 
            Financial Trading"

        Params:
            - end_total_assets (float): total assets after taking an action. Note
            that the transaction fees were subtracted from this value already
            - begin_total_assets (float): total assets before taking the action
            - n_window (int): number of how many steps the agent must have done 
            so that the sharpe ratio is being incorporated into the reward function.
            If this value is None (which is the default value), the sharpe ratio
            is not being considered into the reward
        """
        # Calculate absolute portfolio change
        abs_portfolio_change = end_total_assets - begin_total_assets

        # Check if sharpe ratio should be considered
        if n_window == None:
            # If not return the absolute change in asset values - scale it as
            # done in the paper "Deep reinforcement learning for automated stock 
            # trading: an ensemble strategy"
            return self.REWARD_SCALING * abs_portfolio_change
        else:
            # Sharpe ratio will be considered in this case

            # Get the number of already done actions
            num_actions = len(self.asset_memory) - 1
            
            # In 'A Sharpe Ratio Based Reward Scheme in Deep Reinforcement Learning 
            # for Financial Trading' the P&L was not the absolute portfolio value
            # change but the return. We therefore divide by the initial portfolio 
            # value and need to add the sharpe ratio accordingly. Multiply by 100
            # to get the value in %
            portfolio_change = abs_portfolio_change/begin_total_assets * 100

            # Note that the fee will not be added extra since it is already 
            # incorporated into the 'end_total_assets' (cash position was reduced
            # by the transaction costs)
            
            # Check that at least n-steps were done in the environment 
            if num_actions >= n_window:
                # Calculate the sharpe ratio
                sharpe_ratio_new = MultiStockTrading.calculate_sharpe_ratio(
                                                                self.asset_memory,
                                                                annualize_daily=False
                                                                )

                # If sharpe ratio is bigger than previous one, we add it
                # for num_actions = n_window (t=w), this is always the case, since 
                # self.sharpe_ratio is set to -np.inf
                if sharpe_ratio_new >= self.sharpe_ratio:
                    reward = portfolio_change + sharpe_ratio_new
                else:
                    # Otherwise, we subtract it from the portfolio change
                    reward = portfolio_change - sharpe_ratio_new
                
                # Update sharpe ratio
                self.sharpe_ratio = sharpe_ratio_new

                return reward
            else:
                # If not enough actions done, return the absolute value change
                return portfolio_change

    @staticmethod
    def calculate_sharpe_ratio(vals:list,annualize_daily:bool=True):
        """ 
        For a given list of total assets, this method will calculate
        the sharpe ratio. One can also annualize it. Note that
        by annualizing, the sharpe-ratio is multiplied by sqrt(252).
        Therefore, it is assumed that daily returns are given. Otherwise,
        the annualization should be done differently.

        Params:
            - vals (list): list containing the total assets 
        """
        # Convert the asset memory to a pandas dataframe
        market_data_total_value = pd.DataFrame(vals)
        
        # Calculate the percentage change (return)
        pct_change = market_data_total_value.pct_change()

        # Calculate the sharpe-ratio
        sharpe_ratio = (pct_change.mean()/pct_change.std()).iloc[0]

        # Calculate the annualized sharpe ratio based on the daily returns 
        if annualize_daily:
            return (252**0.5)*sharpe_ratio
        else:
            return sharpe_ratio

    def _get_info(self):
        """ 
        This method will return the current necessary information
        within the environment. More specifically, the trading day,
        the current amount of cash, the absolute number of shares 
        that the agent owns and the prices will be returned as a 
        dictionary.
        """
        return {
            "num trade": self.day,
            "cash":self.state[0],
            "shares":self.state[1:self.STOCK_DIM+1],
            "prices":self.state[self.STOCK_DIM+1:2*self.STOCK_DIM+1]
        }
    
    def reset(self, seed = None, options = None):
        """ 
        This method resets the whole environment.

        Params:
            - seed (int) used for sampling (default None)
            - options (dict) default is None (needed to inherit from
            gymnasium)

        Returns:
            - state (list): initial state of the environment after reset
            - info (dict): additional information 
        """
        # Seed self.np_random
        super().reset(seed=seed)

        # Reset the asset memory, reward memory and terminal variable 
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.terminal = False 

        # Set day to zero (to start from beginning of the dataframe)
        self.day = 0

        # Get the initial market data 
        self.data = self.market_data.loc[self.day,:]
        info = self._get_info()

        # Initialize the state 
        state_tmp = [self.INITIAL_ACCOUNT_BALANCE if i == 0 else 0 
                     for i in range(self.STOCK_DIM+1)]
        self.state = np.concatenate(
                                    [state_tmp, 
                                     self.data,
                                     [self.turbulence_index.iloc[self.day,0]]],
                                     dtype=np.float32)
        
        return self.state, info

    def render(self):
        return self.state
    
    def _get_buy_sell_stocks(self, actions):
        """ 
        For a given vector of actions, this method will return which 
        stocks need to be bought and which need to be sold. Note that the
        stocks will be sorted by the amount of shares to buy/sell. 

        Params:
            - actions (np.array): list containing for each stock the 
              amount of shares to buy/sell

        Returns:
            - buy_index (np.array): list containing the indeces of the stocks
              need to be bought. The values are stored from highest to lowest (e.g.
              [3,1] means that the fourth stock has a higher action value 
              than the second stock)
            - sell_index (np.array): list containing the indeces of the stocks
              need to be bought. The values are stored from lowest to highest (e.g.
              [3,1] means that the fourth stock has a lower action value 
              than the second stock. Thus, one needs to sell more shares of the
              fourth stock than the second.)
        """
        # Get indices of positive values
        positive_indices = np.where(actions > 0)[0]
        negative_indices = np.where(actions < 0)[0]

        # Sort indices based on the values of the array
        sorted_indices_positive = sorted(
                                    positive_indices, 
                                    key = lambda x: actions[x], 
                                    reverse=True
                                    ) 
        
        sorted_indices_negative = sorted(
                                    negative_indices, 
                                    key=lambda x: actions[x]
                                    )
        
        return sorted_indices_positive, sorted_indices_negative