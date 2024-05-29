## News
Caught up in the whirlwind of NeurIPS'24 — this repo might have to wait a bit for its turn in the spotlight!

# CausalExploration
This is the official implementation of the paper "Boosting Efficiency in Task-Agnostic Exploration Through Causal Knowledge", which was accepted for IJCAI-2024. 

## Introduction
The effectiveness of model training heavily relies on the quality of available training resources. However, budget constraints often impose limitations on data collection efforts. To tackle this challenge, we introduce *causal exploration* in this paper, a strategy that leverages the underlying causal knowledge for both data collection and model training. We, in particular, focus on enhancing the sample efficiency and reliability of the world model learning within the domain of task-agnostic reinforcement learning. During the exploration phase, the agent actively selects actions expected to yield causal insights most beneficial for world model training. Concurrently, the causal knowledge is acquired and incrementally refined with the ongoing collection of data. We demonstrate that causal exploration aids in learning accurate world models using fewer data and provide theoretical guarantees for its convergence. Empirical experiments, on both synthetic data and real-world applications, further validate the benefits of causal exploration.

Our key contributions are summarized as:
- In order to enhance the sample efficiency and reliability of model training with causal knowledge, we introduce a novel concept: causal exploration, and focus particularly on the domain of task-agnostic reinforcement learning.
- To efficiently learn and use causal structural constraints, we develop an online method for causal discovery and formulate the world model with explicit structural embeddings. During exploration, we train the dynamics model under a novel weight-sharing-decomposition schema that can avoid additional computational burden.
- Theoretically, we show that, given strong convexity and smoothness assumptions, our approach attains a superior convergence rate compared to non-causal methods. Empirical experiments further demonstrate the robustness of our online causal discovery method and validate the effectiveness of causal exploration across a range of demanding reinforcement learning environments.

## Method Overview
Throughout the process, the agent, guided by policy $\pi_t$, engages in exploration to gather data that are most beneficial for model training. Meanwhile, causal knowledge and the world model are continuously refined with the ongoing data collection.

<img src="./figures/framework.png" width="100%"/>

## Theoretical Analysis
**Assumption 1.** $L_f$ is strongly convex and smooth such that $\exists ~m>0,~M>0$, for any $\boldsymbol{w} \in \text{dom }L_f$, we have: $$MI \succeq \nabla^2 L_f(\boldsymbol{w}) \succeq mI.$$

The following theorem shows a reduced error bound with causal exploration.

**Theorem 1.** Suppose **Assumption 1** holds, and suppose the density of the causal matrix $D$ is $\delta$ and the model is initialized with $\boldsymbol{w_0}$. Then for every optimization step $k$, we have: $$L_f(\boldsymbol{w}^c(k)) - L_f^\star \le \delta^k \left( L_f(\boldsymbol{w}(k)) - L_f^\star\right) \le \frac M2 \left[ \delta  \left( 1- \frac m M \right)\right]^k \parallel \boldsymbol{w_0} - \boldsymbol{w}^\star \parallel^2_2.$$

The first inequality establishes an upper bound for $\xi$ at $ \delta^k $. Given that $0 \le \delta^k \le 1$, this confirms the effectiveness of causal exploration in enhancing efficiency. The subsequent inequality establishes that the training error of causal exploration gradually converges to the optimal value, exhibiting an \textit{exponential} convergence rate characterized by the decay of the factor $\left[ \delta  \left( 1- \frac m M \right)\right]^k$. Moreover, this theorem implies that the advantages of causal exploration are relevant to the sparseness of causal structure. The sparser the causal structure, the faster our method learns. When the causal matrix $D$ is a complete matrix ($\delta = 1$), causal exploration degenerates into non-causal prediction-based exploration. The proof for Theorem 1 is provided in [Supplementary_material.pdf](https://github.com/CMACH508/CausalExploration/tree/main/Supplementary_material.pdf).

## Experiments
**Synthetic Datasets.**
We build our simulated environment following the state space model with controls. When the agent takes an action $\boldsymbol{a_t}$ based on the current state, the environment provides feedback $\boldsymbol{s_{t+1}}$ at the next time. We denote the generative environment as $$\boldsymbol{s_1} \sim \mathcal{N}(\boldsymbol{0}, I), \quad \boldsymbol{s_t} \sim \mathcal{N} (h(\boldsymbol{s_{t-1}},\boldsymbol{a_{t-1}}), \Sigma),$$ where $\Sigma$ is the covariance matrix and $h$ is the mean value as the ground truth transition function implemented by deep neural networks under causal graph $\mathcal{G}$. Specifically, the linear condition consists of a single-layer network, and the nonlinear function is three-layer MLPs with sigmoid activation.

<img src="./figures/synthetic_exp.png" width="100%"/>

To install, run
```
cd ./simulation 
conda env create -f environment.yml
```
or
```
cd ./simulation 
conda create -n your_env_name python=3.7
pip install -r requirements.txt
```
To train, run
```
python main.py
```

**Traffic Signal Control.**
Traffic signal control is an important means of mitigating congestion in traffic management. Compared to using fixed-duration traffic signals, an RL agent learns a policy to determine real-time traffic signal states based on current road conditions. The state observed by the agent at each time consists of five dimensions of information, namely the number of vehicles, queue length, average waiting time in each lane plus current and next traffic signal states. Action here is to decide whether to change the traffic signal state or not. For example, suppose the traffic signal is red at time $t$, if the agent takes action $1$, then it will change to green at the next time $t+1$, otherwise, it will remain red. Following the work in [IntelliLight](https://dl.acm.org/doi/10.1145/3219819.3220096), the traffic environment in our experiment is a three-lane intersection. 

<img src="./figures/traffic_illu.png" width="100%"/>

**MuJoCo Tasks.**
We also evaluate causal exploration on the challenging [MuJoCo](https://mujoco.org) tasks, where the state-action dimensions range from tens (Hopper-v2) to hundreds (Humanoid-v2). Implementation details and more experimental results including the identified causal structures are given in [Supplementary_material.pdf](https://github.com/CMACH508/CausalExploration/tree/main/Supplementary_material.pdf).

<img src="./figures/mujoco_exp.png" width="100%"/>

## Acknowledgements
Our codes are partly based on the following GitHub repository: [IntelliLight](https://github.com/wingsweihua/IntelliLight), [Mujoco-Pytorch](https://github.com/seolhokim/Mujoco-Pytorch/). Thanks for their awesome works.

## Citation