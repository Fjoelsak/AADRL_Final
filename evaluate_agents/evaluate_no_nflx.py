"""
Evaluates the four agents trained without NFLX in the investment universe.

Produces (saved to evaluate_agents/no_nflx/):
  - performance_table.csv        : annual return, Sharpe, Sortino, max drawdown, ...
  - cumulative_returns.pdf       : performance plot vs. DJI / S&P 500
  - portfolio_allocation.pdf     : portfolio weight per stock over time (all 4 agents)
  - ks_test_results.csv          : KS-test: Agent 4 vs. dominant replacement stock
  - return_distribution.pdf      : daily-return distributions Agent 4 vs. dominant stock

Run from repo root or evaluate_agents/:
    python evaluate_agents/evaluate_no_nflx.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless rendering for server execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import empyrical
from scipy.stats import gaussian_kde
import gymnasium as gym
from collections import defaultdict
from gymnasium.envs.registration import register
from scipy.stats import kstest
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

DATA_DIR    = os.path.join(REPO_ROOT, 'data', 'no_nflx')
MODEL_DIR   = os.path.join(REPO_ROOT, 'train_agents', 'training_eval_no_nflx')
BENCH_DIR   = os.path.join(REPO_ROOT, 'evaluate_agents')   # dji.csv / sp500.csv
OUT_DIR     = os.path.join(REPO_ROOT, 'evaluate_agents', 'no_nflx')
os.makedirs(OUT_DIR, exist_ok=True)

from train_agents.util_functions import evaluate_policy, run_policy

# ---------------------------------------------------------------------------
# Constants (mirror train_no_nflx.py)
# ---------------------------------------------------------------------------
SEEDS        = [42, 7, 25, 14]
N_STEP_SHARPE = 299
NUM_AGENTS   = 4
N_EVAL_RUNS  = 3    # episodes per seed for performance evaluation
N_ALLOC_RUNS = 5    # episodes per seed for allocation analysis

# Agent configurations: (consider_sentiment, consider_sharpe, use_lstm)
AGENT_CONFIGS = [
    (False, None,           False),  # Agent 1
    (True,  None,           False),  # Agent 2
    (True,  N_STEP_SHARPE,  False),  # Agent 3
    (True,  N_STEP_SHARPE,  True),   # Agent 4
]

TICKER_TO_SECTOR = {
    'MRK':  'Healthcare',
    'MS':   'Financials',
    'MU':   'Technology',
    'NVDA': 'Technology',
    'M':    'Consumer Discretionary',
    'EBAY': 'Consumer Discretionary',
    'GILD': 'Healthcare',
    'VZ':   'Communication Services',
    'DAL':  'Industrials',
    'JNJ':  'Healthcare',
    'QCOM': 'Technology',
    'KO':   'Consumer Staples',
    'ORCL': 'Technology',
    'FDX':  'Industrials',
    'HD':   'Consumer Discretionary',
    'WFC':  'Financials',
    'BMY':  'Healthcare',
    'LLY':  'Healthcare',
    'CMG':  'Consumer Discretionary',
    'CAT':  'Industrials',
    'FSLR': 'Technology',
    'NOK':  'Technology',
    'LMT':  'Industrials',
    'MCD':  'Consumer Discretionary',
    'MA':   'Financials',
    'EA':   'Communication Services',
    'FCX':  'Materials',
    'GPS':  'Consumer Discretionary',
    'PEP':  'Consumer Staples',
}

# ---------------------------------------------------------------------------
# Register environment
# ---------------------------------------------------------------------------
register(
    id='Trading-v4',
    entry_point='env.multistock_trading_v4:MultiStockTrading'
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading no-NFLX test data ...")
testing_data = pd.read_csv(
    os.path.join(DATA_DIR, 'testing', 'market_data.csv'),
    index_col=0, header=[0, 1]
)
turbulence_testing = pd.read_csv(
    os.path.join(DATA_DIR, 'testing', 'turbulence_index.csv'),
    index_col=0
)
training_data = pd.read_csv(
    os.path.join(DATA_DIR, 'training', 'market_data.csv'),
    index_col=0, header=[0, 1]
)
tickers = testing_data.columns.get_level_values(1).unique().tolist()
print(f"Tickers ({len(tickers)}): {tickers}")

# Align lengths — turbulence index may have one fewer row than market data
min_len = min(len(testing_data), len(turbulence_testing))
testing_data      = testing_data.iloc[:min_len]
turbulence_testing = turbulence_testing.iloc[:min_len]
print(f"Testing rows after alignment: {min_len}")

# ---------------------------------------------------------------------------
# Load benchmark data (DJI, S&P 500) — reuse files from evaluate_agents/
# ---------------------------------------------------------------------------
dji   = pd.read_csv(os.path.join(BENCH_DIR, 'dji.csv'),   header=[0, 1], index_col=0)
sp500 = pd.read_csv(os.path.join(BENCH_DIR, 'sp500.csv'), header=[0, 1], skiprows=[2], index_col=0)

closes_dji   = dji['Close', '^DJI']
closes_sp500 = sp500['Close', '^GSPC']
cum_returns_dji   = closes_dji   / closes_dji.iloc[0]   - 1
cum_returns_sp500 = closes_sp500 / closes_sp500.iloc[0] - 1

# ---------------------------------------------------------------------------
# 1. Performance evaluation — cumulative returns per agent × seed
# ---------------------------------------------------------------------------
print("\n--- Running performance evaluation ---")
mean_asset_value = np.zeros((NUM_AGENTS, len(SEEDS), len(testing_data)))

for idx_seed, seed in enumerate(SEEDS):
    for agent_num, (consider_sentiment, consider_sharpe, use_lstm) in enumerate(AGENT_CONFIGS):
        model_path = os.path.join(
            MODEL_DIR, f'agent_{agent_num + 1}_seed_{seed}', 'best_model.zip'
        )
        print(f"  Loading agent {agent_num + 1}, seed {seed} from {model_path}")

        test_env = gym.make(
            'Trading-v4',
            market_data=testing_data,
            turbulence_index=turbulence_testing,
            consider_sentiments=consider_sentiment,
            n_step_sharpe=consider_sharpe,
            consider_indicators=[],
        )

        if use_lstm:
            agent = RecurrentPPO.load(model_path, env=test_env)
        else:
            agent = PPO.load(model_path, env=test_env)

        asset_memory, _, _, _, _ = evaluate_policy(
            test_env, agent, n_eval_episodes=N_EVAL_RUNS, deterministic=False
        )
        mean_asset_value[agent_num, idx_seed, :] = np.array(asset_memory).mean(axis=0)

cum_returns = mean_asset_value / mean_asset_value[:, :, 0][:, :, np.newaxis] - 1
mean_cum_returns = cum_returns.mean(axis=1)
std_cum_returns  = cum_returns.std(axis=1)
agent_values_mean = mean_asset_value.mean(axis=1)

# ---------------------------------------------------------------------------
# 2. Performance statistics table
# ---------------------------------------------------------------------------
print("\n--- Computing performance statistics ---")
asset_histories = {f'Agent {i+1}': agent_values_mean[i] for i in range(NUM_AGENTS)}
asset_histories['DJI']    = closes_dji.values[:len(testing_data)]
asset_histories['S&P500'] = closes_sp500.values[:len(testing_data)]

statistics = defaultdict()
for name, history in asset_histories.items():
    pct = np.diff(history) / history[:-1]
    s   = pd.Series(pct, index=pd.to_datetime(testing_data.index[1:]))
    statistics[name] = {
        'Annual return':         empyrical.annual_return(s),
        'Total return':          empyrical.cum_returns_final(s),
        'Annual volatility':     empyrical.annual_volatility(s),
        'Max drawdown':          -empyrical.max_drawdown(s),
        'Sharpe ratio':          empyrical.sharpe_ratio(s),
        'Sortino ratio':         empyrical.sortino_ratio(s),
        'Calmar ratio':          empyrical.calmar_ratio(s),
        'VaR (95%)':             empyrical.value_at_risk(s, cutoff=0.05),
        'VaR (99%)':             empyrical.value_at_risk(s, cutoff=0.01),
        'CVaR (95%)':            empyrical.conditional_value_at_risk(s, cutoff=0.05),
        'CVaR (99%)':            empyrical.conditional_value_at_risk(s, cutoff=0.01),
    }

stats_df = pd.DataFrame(statistics).round(4)
stats_path = os.path.join(OUT_DIR, 'performance_table.csv')
stats_df.to_csv(stats_path)
print(f"  Saved: {stats_path}")
print(stats_df.to_string())

# ---------------------------------------------------------------------------
# 3. Cumulative return plot
# ---------------------------------------------------------------------------
print("\n--- Plotting cumulative returns ---")
fig, ax = plt.subplots(figsize=(7.0, 4))

for agent_num in range(NUM_AGENTS):
    ax.plot(pd.to_datetime(testing_data.index),
            mean_cum_returns[agent_num], label=f'Agent {agent_num + 1}')
    ax.fill_between(pd.to_datetime(testing_data.index),
                    mean_cum_returns[agent_num] - std_cum_returns[agent_num],
                    mean_cum_returns[agent_num] + std_cum_returns[agent_num],
                    alpha=0.2)

ax.plot(pd.to_datetime(cum_returns_dji.index),   cum_returns_dji,   label='DJI')
ax.plot(pd.to_datetime(cum_returns_sp500.index), cum_returns_sp500, label='S&P 500')

ax.set_ylabel('Standardized return', fontsize=8)
ax.set_xlabel('Time', fontsize=8)
ax.legend(fontsize=8)
plt.xticks(rotation=45, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.tight_layout()
perf_path = os.path.join(OUT_DIR, 'cumulative_returns.pdf')
plt.savefig(perf_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {perf_path}")

# ---------------------------------------------------------------------------
# 4. Portfolio allocation
# ---------------------------------------------------------------------------
print("\n--- Running allocation analysis ---")
shares_over_seeds = np.zeros((NUM_AGENTS, len(SEEDS), N_ALLOC_RUNS,
                               len(testing_data) - 1, len(tickers)))
prices_over_seeds = np.zeros_like(shares_over_seeds)
cash_over_seeds   = np.zeros((NUM_AGENTS, len(SEEDS), N_ALLOC_RUNS,
                               len(testing_data) - 1))

for seed_idx, seed in enumerate(SEEDS):
    for agent_num, (consider_sentiment, consider_sharpe, use_lstm) in enumerate(AGENT_CONFIGS):
        model_path = os.path.join(
            MODEL_DIR, f'agent_{agent_num + 1}_seed_{seed}', 'best_model.zip'
        )
        print(f"  Allocation: agent {agent_num + 1}, seed {seed}")

        test_env = gym.make(
            'Trading-v4',
            market_data=testing_data,
            turbulence_index=turbulence_testing,
            consider_sentiments=consider_sentiment,
            n_step_sharpe=consider_sharpe,
            consider_indicators=[],
        )
        if use_lstm:
            agent = RecurrentPPO.load(model_path)
        else:
            agent = PPO.load(model_path)

        shares, prices, cash = run_policy(test_env, agent, n_eval_episodes=N_ALLOC_RUNS)
        shares_over_seeds[agent_num, seed_idx] = shares
        prices_over_seeds[agent_num, seed_idx] = prices
        cash_over_seeds[agent_num, seed_idx]   = cash

# Mean over seeds and runs
mean_shares = shares_over_seeds.mean(axis=2).mean(axis=1)
mean_cash   = cash_over_seeds.mean(axis=2).mean(axis=1)
prices_arr  = prices_over_seeds[0, 0]    # prices are deterministic

mean_shares_dfs = [
    pd.DataFrame(mean_shares[i], columns=tickers,
                 index=pd.to_datetime(testing_data.index[1:]))
    for i in range(NUM_AGENTS)
]

# Weight shares by price → portfolio value per stock
mean_shares_cash = [df.copy() for df in mean_shares_dfs]
weighted_num     = [df.copy() for df in mean_shares_dfs]
weighted_pct     = [df.copy() for df in mean_shares_dfs]

for i, df in enumerate(mean_shares_dfs):
    mean_shares_cash[i]['Cash'] = mean_cash[i]
    combined = np.concatenate([prices_arr, np.ones((len(prices_arr), 1))], axis=1)
    weighted_num[i]  = mean_shares_cash[i] * combined
    weighted_pct[i]  = weighted_num[i].div(weighted_num[i].sum(axis=1), axis=0) * 100

# Portfolio allocation plot (per stock, weighted)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 5), sharex=True, sharey=True)
col, row = 0, 0
palette = [cm.tab20(i % 20) for i in range(len(tickers) + 1)]

for i in range(NUM_AGENTS):
    weighted_pct[i].plot(
        kind='area', stacked=True, alpha=0.8,
        ax=ax[row][col], color=palette, legend=(i == 0)
    )
    if i == 0:
        ax[row][col].legend(loc='upper center', fontsize=7,
                            bbox_to_anchor=(1, -1.6),
                            ncol=len(weighted_pct[i].columns) // 3)
    else:
        ax[row][col].get_legend().remove()

    ax[row][col].set_title(f'Agent {i + 1}')
    ax[row][col].tick_params(axis='x', labelsize=9, rotation=45)
    col += 1
    if col == 2:
        row += 1
        col  = 0

ax[1][0].set_xlabel('Time')
ax[1][1].set_xlabel('Time')
ax[0][0].set_ylabel('Portfolio weight (%)')
ax[1][0].set_ylabel('Portfolio weight (%)')
plt.subplots_adjust(wspace=0.03, hspace=0.15)

alloc_path = os.path.join(OUT_DIR, 'portfolio_allocation.pdf')
plt.savefig(alloc_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {alloc_path}")

# Print dominant stock weights at end of period for each agent
print("\nTop-5 portfolio weights at end of test period:")
for i in range(NUM_AGENTS):
    top5 = weighted_pct[i].iloc[-1].nlargest(5)
    print(f"  Agent {i + 1}: {top5.round(2).to_dict()}")

# ---------------------------------------------------------------------------
# 5. KS test: Agent 4 vs. dominant replacement stock
# ---------------------------------------------------------------------------
print("\n--- KS test: Agent 4 vs. dominant stock ---")
dominant_stock = weighted_pct[3].iloc[-1].drop('Cash', errors='ignore').idxmax()
print(f"  Dominant stock in Agent 4 (no NFLX): {dominant_stock}")

closes_dominant = testing_data['Adj Close'][dominant_stock]
daily_returns_agent4   = np.ravel(pd.DataFrame(agent_values_mean[3]).pct_change().dropna())
daily_returns_dominant = np.ravel(pd.DataFrame(closes_dominant).pct_change().dropna())

ks_result = kstest(daily_returns_agent4, daily_returns_dominant)
print(f"  KS statistic = {ks_result.statistic:.4f},  p-value = {ks_result.pvalue:.2e}")

ks_df = pd.DataFrame({
    'comparison':   [f'Agent 4 vs. {dominant_stock}'],
    'ks_statistic': [ks_result.statistic],
    'p_value':      [ks_result.pvalue],
    'dominant_stock': [dominant_stock],
})
ks_path = os.path.join(OUT_DIR, 'ks_test_results.csv')
ks_df.to_csv(ks_path, index=False)
print(f"  Saved: {ks_path}")

# ---------------------------------------------------------------------------
# 6. Return distribution plot: Agent 4 vs. dominant stock
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
for returns, label, color in [
    (daily_returns_dominant, dominant_stock,  'red'),
    (daily_returns_agent4,   'Agent 4',       'cornflowerblue'),
]:
    ax.hist(returns, bins=50, density=True, alpha=0.4, color=color, label=label)
    x = np.linspace(returns.min(), returns.max(), 300)
    kde = gaussian_kde(returns)
    ax.plot(x, kde(x), color=color)
ax.set_xlabel('Daily return')
ax.set_xlim(-0.05, 0.05)
ax.set_ylabel('Density')
ax.legend()
ax.set_title(f'Return distribution: Agent 4 vs. {dominant_stock} (no NFLX)')
plt.tight_layout()
dist_path = os.path.join(OUT_DIR, 'return_distribution.pdf')
plt.savefig(dist_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {dist_path}")

print("\nDone. All outputs in:", OUT_DIR)
