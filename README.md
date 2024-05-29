# CausalExploration
This is the official implementation of the paper "Boosting Efficiency in Task-Agnostic Exploration Through Causal Knowledge", which was accepted for IJCAI-2024. 

## Introduction
The effectiveness of model training heavily relies on the quality of available training resources. However, budget constraints often impose limitations on data collection efforts. To tackle this challenge, we introduce *causal exploration* in this paper, a strategy that leverages the underlying causal knowledge for both data collection and model training. We, in particular, focus on enhancing the sample efficiency and reliability of the world model learning within the domain of task-agnostic reinforcement learning. During the exploration phase, the agent actively selects actions expected to yield causal insights most beneficial for world model training. Concurrently, the causal knowledge is acquired and incrementally refined with the ongoing collection of data. We demonstrate that causal exploration aids in learning accurate world models using fewer data and provide theoretical guarantees for its convergence. Empirical experiments, on both synthetic data and real-world applications, further validate the benefits of causal exploration.

Our key contributions are summarized as:
- In order to enhance the sample efficiency and reliability of model training with causal knowledge, we introduce a novel concept: causal exploration, and focus particularly on the domain of task-agnostic reinforcement learning.
- To efficiently learn and use causal structural constraints, we develop an online method for causal discovery and formulate the world model with explicit structural embeddings. During exploration, we train the dynamics model under a novel weight-sharing-decomposition schema that can avoid additional computational burden.
- Theoretically, we show that, given strong convexity and smoothness assumptions, our approach attains a superior convergence rate compared to non-causal methods. Empirical experiments further demonstrate the robustness of our online causal discovery method and validate the effectiveness of causal exploration across a range of demanding reinforcement learning environments.

## News
Caught up in the whirlwind of NeurIPS'24 — this repo might have to wait a bit for its turn in the spotlight!