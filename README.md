# Code implementation of Data-driven strategy for Wordle

This repository contains the code and documentation for my MSc Data Science Extended Research Project. The project explores and compares three distinct approaches to solving the Wordle puzzle:

1.  **Heuristic Baseline (`heuristic_solver.py`)**: A rule-based solver using letter and position frequency analysis.
2.  **Entropy-Optimized Solver (`entropy_solver.py`)**: An information-theoretic approach that maximizes information gain, featuring a second-guess lookup table.
3.  **Reinforcement Learning Solver (`rl_solver.py`)**: A Dueling Double DQN agent trained with Behavioral Cloning and Conservative Q-Learning (CQL).

## Quick Results Summary

| Solver | Average Attempts (95% CI) | Success Rate (95% CI) | Key Feature |
| :--- | :--- | :--- | :--- |
| **Heuristic Baseline** | ~3.80 | ~98.4% | Letter frequency scoring |
| **Entropy-Optimized** | ~3.68 | ~98.6% | Second-guess lookup table |
| **RL (Ours)** | ~3.64| ~99.6%| Dueling-DDQN + BC + CQL |

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip or conda

### Installation

1.  **Clone the repository**

2.  **Set up the environment (using Conda is recommended)**
    ```bash
    # Option A: Using conda (recommended)
    conda env create -f environment.yml
    conda activate wordle-solver

    # Option B: Using pip
    pip install -r requirements.txt
    ```

### Data Preparation

The solvers require two text files representing the Wordle word lists:

- `wordlist_hidden.txt` — answer list (five-letter words)  
- `wordlist_all.txt` — allowed guess list (five-letter words)  
- Once obtained, place both files under the `./data/` directory, keeping the exact filenames above.  

## How to Run

### 1. Heuristic Baseline Solver
Run 3000 simulations and generate performance plots.
```bash
python src/heuristic_solver.py
```
*Outputs will be saved in the `figs_baseline/` directory.*

### 2. Entropy-Optimized Solver
Run 5000 simulations with second-guess optimization.
```bash
python src/entropy_solver.py
```
*Outputs will be saved in the `figs_entropy/` directory.*

### 3. Reinforcement Learning Solver

#### To Train the Model (Time-consuming)
This will train the model from scratch on 30,000 demonstration games.
```bash
python src/rl_solver.py
```
*The trained model will be saved as `wordle_dqn_enhanced.pth`.*

#### To Evaluate a Pre-trained Model
We provide a pre-trained model.
```bash
python src/rl_solver.py --eval-only --model-path path/to/wordle_dqn_enhanced.pth
```
*Outputs will be saved in the `figs_rl/` directory and results printed to the console.*


## Dependencies

Core dependencies are listed in `requirements.txt` and `environment.yml`. The main libraries include:
- `torch` (PyTorch for deep learning)
- `numpy` (numerical computations)
- `matplotlib` (plotting results)
- `wordfreq` (for word frequency data)
