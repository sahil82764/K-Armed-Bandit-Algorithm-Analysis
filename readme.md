# Multi-Armed Bandit Learning Algorithms

**Student:** Sahil Khan  
**Course:** DSCI 6650-001: Reinforcement Learning  
**Date:** June 18, 2025  
**Assignment:** k-Armed Bandit Learning Algorithms

## Problem Statement

The multi-armed bandit problem represents a fundamental challenge in reinforcement learning where an agent must repeatedly choose between multiple actions (arms) to maximize cumulative reward over time. Each arm provides stochastic rewards from unknown distributions, creating a classic exploration-exploitation dilemma: should the agent exploit the currently best-known arm or explore other arms that might yield higher rewards?

This study addresses the bandit problem in both stationary environments (where reward distributions remain constant) and non-stationary environments (where reward distributions change over time). The research investigates how different algorithms handle gradual changes (drift and mean-reversion) and abrupt changes (sudden permutation of reward means) in the environment.

## Overview

This repository contains a comprehensive implementation and analysis of four multi-armed bandit algorithms across stationary and non-stationary environments. The study evaluates different exploration-exploitation strategies using a 10-armed testbed and provides insights into algorithm selection based on environmental characteristics.

## Algorithms Implemented

1. **Greedy Algorithm** (ε = 0): Pure exploitation with no exploration
2. **Epsilon-Greedy Algorithm**: Balanced exploration-exploitation with ε-probability exploration
3. **Optimistic Initial Values**: Encourages early exploration through optimistic initialization
4. **Gradient Bandit Algorithm**: Uses action preferences and softmax selection

## Experimental Setup

- **Environment**: 10-armed bandit testbed
- **Reward Distribution**: N(μᵢ, 1) where μᵢ ~ N(0, 1)
- **Time Steps**: 2000 per simulation
- **Simulations**: 1000 independent runs
- **Random Seed**: 42 (for reproducibility)

## Files Description

- `BANDIT_LEARNING_ALGORITHM.ipynb`: Complete implementation with all experiments
- `ProjectReport.docx`: Comprehensive analysis report with results and visualizations
- `a1.pdf`: Original assignment instructions

## Part 1: Stationary Environment

Compares all four algorithms in a stationary 10-armed bandit environment. Key findings:
- **Optimistic Initial Values** performed best overall
- **Gradient Bandit** showed strong performance with good exploration-exploitation balance
- **Epsilon-Greedy** demonstrated moderate but consistent performance
- **Greedy** performed poorly due to lack of exploration

## Part 2: Non-Stationary Environments

### Part 2.1: Gradual Changes
Tests algorithms under two non-stationary conditions:
- **Drift Model**: μᵢ,ₜ = μᵢ,ₜ₋₁ + εᵢ,ₜ where εᵢ,ₜ ~ N(0, 0.01²)
- **Mean-Reverting Model**: μᵢ,ₜ = 0.5μᵢ,ₜ₋₁ + εᵢ,ₜ where εᵢ,ₜ ~ N(0, 0.01²)

### Part 2.2: Abrupt Changes
Evaluates performance when means are randomly permuted at t = 501:
- Compares "as-is" performance vs. hard reset scenarios
- **Gradient Bandit** showed superior adaptability to abrupt changes

## Hyperparameter Tuning

Optimal parameters found through pilot experiments:
- **Epsilon-Greedy**: ε = 0.05
- **Gradient Bandit**: α = 0.05
- **Optimistic Initial Value**: 4.128 (99.5th percentile of highest mean)

## Key Results

- **Stationary environments**: Optimistic initialization excels when prior knowledge is available
- **Non-stationary environments**: Gradient bandit methods provide superior adaptation
- **Exploration is crucial**: Even simple exploration strategies significantly improve performance

## Requirements

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```

## Usage

1. Clone the repository
2. Open `BANDIT_LEARNING_ALGORITHM.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells to reproduce the complete experiment
4. Results include both numerical outputs and visualization plots

## Reproducibility

All experiments use fixed random seeds (seed=42) ensuring reproducible results across runs. The implementation follows best practices for reinforcement learning experimentation with proper statistical averaging and comprehensive visualization.

## Results Visualization

The notebook generates multiple plots showing:
- Average reward over time for each algorithm
- Percentage of optimal actions selected
- Comparative performance across different environmental conditions
- Hyperparameter tuning results

## Citation

If you use this code or analysis, please reference:
```
Khan, S. (2025). Multi-Armed Bandit Learning Algorithms Analysis. 
DSCI 6650-001: Reinforcement Learning Assignment 1.
```