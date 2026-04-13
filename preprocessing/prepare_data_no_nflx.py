"""
Prepares the agent-environment data and turbulence index with NFLX excluded
from the investment universe. Outputs are saved to data/no_nflx/ and split
into train/validation/test subfolders mirroring the original data layout.
"""

import sys
import os
import numpy as np
import pandas as pd

# Allow imports from repo root regardless of working directory
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
from preprocessing.preprocessor import Preprocessor

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
MARKET_DATA_PATH    = os.path.join(REPO_ROOT, 'data', 'market_data.csv')
SA_DATA_PATH        = os.path.join(REPO_ROOT, 'data', 'stock_sentiments_final.csv')
OUTPUT_DIR          = os.path.join(REPO_ROOT, 'data', 'no_nflx')

STARTING_DATE_VALIDATION = '2014-10-01'
STARTING_DATE_TEST       = '2016-01-04'

EXCLUDE_TICKER = 'NFLX'

# ---------------------------------------------------------------------------
# Load raw data and drop NFLX
# ---------------------------------------------------------------------------
print(f"Loading market data from {MARKET_DATA_PATH} ...")
market_data = pd.read_csv(MARKET_DATA_PATH, header=[0, 1], index_col=0)

# Drop all columns belonging to the excluded ticker across every price type
nflx_cols = [(lvl0, lvl1)
             for lvl0, lvl1 in market_data.columns
             if lvl1 == EXCLUDE_TICKER]
market_data_no_nflx = market_data.drop(columns=nflx_cols)
print(f"Removed {len(nflx_cols)} columns for {EXCLUDE_TICKER}. "
      f"Remaining tickers: {market_data_no_nflx.columns.get_level_values(1).unique().tolist()}")

print(f"Loading sentiment data from {SA_DATA_PATH} ...")
sentiments = pd.read_csv(SA_DATA_PATH, index_col=0)
sentiments_no_nflx = sentiments[sentiments['stock'] != EXCLUDE_TICKER].copy()
print(f"Removed {len(sentiments) - len(sentiments_no_nflx)} sentiment rows for {EXCLUDE_TICKER}.")

# ---------------------------------------------------------------------------
# Preprocess (technical indicators + merge with sentiments)
# ---------------------------------------------------------------------------
print("Running Preprocessor ...")
preprocessor = Preprocessor(market_data_no_nflx, sentiments_no_nflx)
agent_env_data = preprocessor.get_preprocessed_data()

assert not agent_env_data.isna().any(axis=None), \
    "Preprocessed data contains NaN values — check the Preprocessor output."

# ---------------------------------------------------------------------------
# Calculate turbulence index (same method as original create_env.ipynb)
# ---------------------------------------------------------------------------
print("Calculating turbulence index ...")
close_prices = agent_env_data['Adj Close']

# Split at the test-set boundary to calibrate only on training history
start_test_idx = close_prices.index.get_loc(
    close_prices.index[close_prices.index == STARTING_DATE_TEST][0]
)
training_close = close_prices.iloc[:start_test_idx]
test_close     = close_prices.iloc[start_test_idx:]

training_returns = training_close.pct_change().dropna()
test_returns     = test_close.pct_change()

cov_training   = training_returns.cov()
means_training = training_returns.mean()

def calc_turbulence(r):
    diff = r - means_training
    return float(diff @ np.linalg.inv(cov_training) @ diff)

turbulences_training = training_returns.apply(calc_turbulence, axis=1)
turbulences_test     = test_returns.apply(calc_turbulence, axis=1)
turbulences          = pd.concat([turbulences_training, turbulences_test])
turbulences          = pd.DataFrame({'Turbulences': turbulences})
turbulences.iloc[0]  = 0  # first value is NaN from pct_change; set to 0

# ---------------------------------------------------------------------------
# Train / Validation / Test split (mirrors original logic)
# ---------------------------------------------------------------------------
val_start  = agent_env_data.index[agent_env_data.index == STARTING_DATE_VALIDATION][0]
test_start = agent_env_data.index[agent_env_data.index == STARTING_DATE_TEST][0]

training_data   = agent_env_data.loc[:val_start].iloc[:-1]
validation_data = agent_env_data.loc[val_start:test_start].iloc[:-1]
testing_data    = agent_env_data.loc[test_start:]

turbulence_training   = turbulences.iloc[:len(training_data)]
turbulence_validation = turbulences.iloc[len(training_data):len(training_data) + len(validation_data)]
turbulence_testing    = turbulences.iloc[len(training_data) + len(validation_data):]

print(f"Split sizes — train: {len(training_data)}, "
      f"val: {len(validation_data)}, test: {len(testing_data)}")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ['training', 'validation', 'testing']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

agent_env_data.to_csv(os.path.join(OUTPUT_DIR, 'agent_environment_data.csv'))
turbulences.to_csv(os.path.join(OUTPUT_DIR, 'turbulence_index.csv'))

training_data.to_csv(os.path.join(OUTPUT_DIR, 'training',   'market_data.csv'))
validation_data.to_csv(os.path.join(OUTPUT_DIR, 'validation', 'market_data.csv'))
testing_data.to_csv(os.path.join(OUTPUT_DIR, 'testing',     'market_data.csv'))

turbulence_training.to_csv(os.path.join(OUTPUT_DIR, 'training',   'turbulence_index.csv'))
turbulence_validation.to_csv(os.path.join(OUTPUT_DIR, 'validation', 'turbulence_index.csv'))
turbulence_testing.to_csv(os.path.join(OUTPUT_DIR, 'testing',     'turbulence_index.csv'))

print(f"All files written to {OUTPUT_DIR}/")
