# Stochastic Topology and Risk-Aware Control

### A Comparative Analysis of On-Policy vs. Off-Policy Reinforcement Learning
**Course:** CAP 6629 | **Date:** Fall 2025

---

## 📌 Project Overview
This research investigates the "Safety vs. Optimality" trade-off in Reinforcement Learning agents. By deploying **SARSA**, **Q-Learning**, and **Double Q-Learning** in a custom stochastic grid environment ($13 \times 13$), we analyze how different temporal difference (TD) learning targets affect an agent's behavior in the presence of catastrophic risks (cliffs) and environmental uncertainty (wind/slip).

### 🧪 Key Experiments
1.  **Theoretical Ground Truth:** Calculated $V^*(s)$ using Model-Based Value Iteration.
2.  **Convergence Analysis:** Compared Mean Squared Error (MSE) of learning agents against the ground truth.
3.  **Risk Quantification:** Visualized internal Q-values to map "fear" zones in stochastic regions.

---

## 🚀 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute Experiment:**
    ```bash
    cd src
    python main.py
    ```
3.  **View Results:**
    * Plots are saved to `/assets`
    * Raw Data is saved to `/data`

---

## 📂 Repository Structure
* `src/environment.py`: Custom Gym-style grid world with Wind & Slip dynamics.
* `src/algorithms.py`: Implementations of SARSA, Q-Learning, Double Q, and VI.
* `src/visualization.py`: Tools for heatmap generation and metric plotting.# Risk-Aware-Stochastic-Navigation
