# common library
import pandas as pd
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import os

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "done_data.csv"
    """
    ### Commented out to add the turbulence index ###
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)
    """    
    for idx_seed,seed in enumerate([42,7,25,14]): ## added later to have multiple seeds 
        # Read the data and add the turbulence index 
        # Keep the technical indicators unchanged!
        data = pd.read_csv(preprocessed_path, index_col=0)
        
        # Add the turbulence index for the first run
        if idx_seed == 0:
            data = add_turbulence(data)
            data.to_csv(preprocessed_path)

        print(data.head())
        print(data.size)


        # 2015/10/01 is the date that validation starts
        # 2016/01/01 is the date that real trading starts
        # unique_trade_date needs to start from 2015/10/01 for validation purpose
        #unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
        unique_trade_date = data[(data.datadate > 20151001)].datadate.unique() # remove the upper limit
        print(unique_trade_date)

        # rebalance_window is the number of months to retrain the model
        # validation_window is the number of months to validation the model and select for trading
        rebalance_window = 63
        validation_window = 63
        
        # Create results directiory (results will be stored there)
        directory_path = "results"
        os.mkdir(directory_path)

        ## Ensemble Strategy
        run_ensemble_strategy(df=data, 
                            unique_trade_date= unique_trade_date,
                            rebalance_window = rebalance_window,
                            validation_window=validation_window,
                            seed=seed) ## seed added later for reproducibility

        # Rename the directoy to the corresponding seed
        new_directory_name = f"results_seed_{seed}"
        os.rename(directory_path, new_directory_name)


        #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
