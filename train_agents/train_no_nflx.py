"""
Trains agents 1-4 without NFLX in the investment universe.
Uses the same hyperparameters and seeds as the original training.
Results are saved to train_agents/training_eval_no_nflx/.

Agent configurations:
  Agent 1 — MLP,       no sentiment,  absolute portfolio change reward
  Agent 2 — MLP,       sentiment,     absolute portfolio change reward
  Agent 3 — MLP,       sentiment,     Sharpe-ratio reward
  Agent 4 — MLP+LSTM,  sentiment,     Sharpe-ratio reward

Run this script from train_agents/ or from the repo root:
    python train_agents/train_no_nflx.py
"""

import sys
import os
import copy
import pandas as pd
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

# Allow imports from repo root regardless of working directory
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
from train_agents.eval_callback import EvaluationCallback

# ---------------------------------------------------------------------------
# Paths and constants (relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR         = os.path.join(REPO_ROOT, 'data', 'no_nflx')
LOG_DIR          = os.path.join(REPO_ROOT, 'train_agents', 'training_eval_no_nflx')
BEST_PARAMS_CSV  = os.path.join(REPO_ROOT, 'train_agents', 'training_eval', 'best_params_hyperparameteropt.csv')

NUMBER_ENVS      = 4
N_STEP_VAL       = 2_500
N_EPISODE_VAL    = 3
TIMESTEPS        = 5e5
N_STEP_SHARPE    = 299

RANDOM_SEEDS = [42, 7, 25, 14]

# Agent configurations: (consider_sentiment, consider_sharpe, use_lstm)
AGENT_CONFIGS = [
    (False, None,          False),   # Agent 1
    (True,  None,          False),   # Agent 2
    (True,  N_STEP_SHARPE, False),   # Agent 3
    (True,  N_STEP_SHARPE, True),    # Agent 4
]

# ---------------------------------------------------------------------------
# Register trading environment
# ---------------------------------------------------------------------------
register(
    id='Trading-v4',
    entry_point='env.multistock_trading_v4:MultiStockTrading'
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading no-NFLX data ...")
agent_env_data = pd.read_csv(
    os.path.join(DATA_DIR, 'agent_environment_data.csv'),
    index_col=0, header=[0, 1]
)
turbulence_index = pd.read_csv(
    os.path.join(DATA_DIR, 'turbulence_index.csv'),
    index_col=0
)
turbulence_index = turbulence_index.fillna(0)

training_data = pd.read_csv(
    os.path.join(DATA_DIR, 'training', 'market_data.csv'),
    index_col=0, header=[0, 1]
)
validation_data = pd.read_csv(
    os.path.join(DATA_DIR, 'validation', 'market_data.csv'),
    index_col=0, header=[0, 1]
)

turbulence_training = pd.read_csv(
    os.path.join(DATA_DIR, 'training', 'turbulence_index.csv'),
    index_col=0
)
turbulence_validation = pd.read_csv(
    os.path.join(DATA_DIR, 'validation', 'turbulence_index.csv'),
    index_col=0
)

print(f"Training rows: {len(training_data)}, "
      f"Validation rows: {len(validation_data)}")

# ---------------------------------------------------------------------------
# Load best hyperparameters (from original hyperparameter optimisation)
# ---------------------------------------------------------------------------
bp = pd.read_csv(BEST_PARAMS_CSV).iloc[0]

best_params = {
    'batch_size':    int(bp['batch_size']),
    'n_steps':       int(bp['n_steps']),
    'gamma':         float(bp['gamma']),
    'learning_rate': float(bp['learning_rate']),
    'clip_range':    float(bp['clip_range']),
    'n_epochs':      int(bp['n_epochs']),
    'gae_lambda':    float(bp['gae_lambda']),
    'max_grad_norm': float(bp['max_grad_norm']),
    'policy_kwargs': {
        'net_arch': dict(pi=[64, 64], vf=[64, 64])
        if bp['net_arch'] == 'small'
        else dict(pi=[256, 64], vf=[256, 64])
    }
}

recurrent_params = copy.deepcopy(best_params)
recurrent_params['policy_kwargs']['lstm_hidden_size'] = int(bp['lstm_hidden_size'])

print(f"Loaded hyperparameters: {best_params}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

for seed in RANDOM_SEEDS:
    for agent_num, (consider_sentiment, consider_sharpe, use_lstm) in enumerate(AGENT_CONFIGS):
        agent_label = f'agent_{agent_num + 1}_seed_{seed}'
        save_path   = os.path.join(LOG_DIR, agent_label)

        print(f"\n{'='*60}")
        print(f"Training {agent_label}  |  sentiment={consider_sentiment}  "
              f"sharpe_window={consider_sharpe}  lstm={use_lstm}")
        print(f"{'='*60}")

        env = make_vec_env(
            'Trading-v4',
            n_envs=NUMBER_ENVS,
            env_kwargs={
                'market_data':       training_data,
                'turbulence_index':  turbulence_training,
                'consider_sentiments': consider_sentiment,
                'n_step_sharpe':     consider_sharpe,
            }
        )

        eval_env = gym.make(
            'Trading-v4',
            market_data=validation_data,
            turbulence_index=turbulence_validation,
            consider_sentiments=consider_sentiment,
            n_step_sharpe=consider_sharpe,
        )

        if use_lstm:
            model = RecurrentPPO(
                'MlpLstmPolicy',
                env,
                seed=seed,
                verbose=0,
                **recurrent_params
            )
        else:
            model = PPO(
                'MlpPolicy',
                env,
                seed=seed,
                verbose=0,
                **best_params
            )

        callback = EvaluationCallback(
            eval_env,
            eval_freq=N_STEP_VAL * NUMBER_ENVS,
            n_eval_episodes=N_EPISODE_VAL,
            save_every_nstep=True,
            saving_freq=10_000,
            log_path=save_path,
            verbose=1,
            additional_name_eval_file='validation'
        )

        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callback,
            progress_bar=True
        )

        print(f"Finished {agent_label}. Best model saved to {save_path}/")

print("\nAll agents trained successfully.")
