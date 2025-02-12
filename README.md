Autonomous Driving with SAC and LSTM Trajectory Prediction
Project Overview
Implementation of an autonomous driving agent using Soft Actor-Critic (SAC) and LSTM-based trajectory prediction in MetaDrive.

Components:
sac_env.py: SAC training environment setup
traj_collector.py: Trajectory data collection from trained agent
traj_predictor.py: LSTM trajectory prediction and visualization
evaluate_agent.py: Performance evaluation metrics

Key Features:
SAC agent trained on 100 unique traffic scenarios
LSTM prediction (10-step input → 30-step prediction)
Trajectory visualization via GIF generation
85% success rate in test environments

Requirements:
Python 3.7+
PyTorch
MetaDrive
Stable-baselines3
Matplotlib
NumPy

Usage:
Train SAC agent: python sac_env.py --train
Collect trajectories: python traj_collector.py
Train predictor and generate visualizations: python traj_predictor.py
Evaluate agent: python evaluate_agent.py

Results:
Success Rate: 85%
Safety Violations: 0.17 ±0.38
Average Return: 215.61 ±28.99
