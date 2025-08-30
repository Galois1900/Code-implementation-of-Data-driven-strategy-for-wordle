# Code-implementation-of-Data-driven-strategy-for-wordle
ERP reproducibility pack
# MSc Data Science Research Project: Advanced Wordle Solvers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and documentation for my MSc Data Science Extended Research Project. The project explores and compares three distinct approaches to solving the Wordle puzzle:

1.  **Heuristic Baseline (`heuristic_solver.py`)**: A rule-based solver using letter and position frequency analysis.
2.  **Entropy-Optimized Solver (`entropy_solver.py`)**: An information-theoretic approach that maximizes information gain, featuring a second-guess lookup table.
3.  **Reinforcement Learning Solver (`rl_solver.py`)**: A advanced Dueling Double DQN agent trained with Behavioral Cloning and Conservative Q-Learning (CQL).

## 📊 Quick Results Summary

| Solver | Average Attempts (95% CI) | Success Rate (95% CI) | Key Feature |
| :--- | :--- | :--- | :--- |
| **Heuristic Baseline** | ~3.85 | ~98.5% | Letter frequency scoring |
| **Entropy-Optimized** | ~3.62 | ~99.2% | Second-guess lookup table |
| **RL (Ours)** | **~3.48** | **~99.6%** | **Dueling-DDQN + BC + CQL** |

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   pip or conda

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up the environment (using Conda is recommended)**
    ```bash
    # Option A: Using conda (recommended)
    conda env create -f environment.yml
    conda activate wordle-solver

    # Option B: Using pip
    pip install -r requirements.txt
    ```

### Data Preparation

The solvers require the official Wordle word lists.
1.  Download the answer list (`wordlist_nyt20230701_hidden.txt`) and the allowed guess list (`wordlist_nyt20220830_all.txt`).
2.  Place these two files in the `data/` directory.

*(Note: Due to copyright, the word lists cannot be distributed here. A quick web search will help you find them.)*

## 🧪 How to Run

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
We provide a pre-trained model. Contact me for access.
```bash
python src/rl_solver.py --eval-only --model-path path/to/wordle_dqn_enhanced.pth
```
*Outputs will be saved in the `figs_rl/` directory and results printed to the console.*

## 📁 Project Structure

```
msc-wordle-project/
├── data/               # Word lists directory (see 'Data Preparation')
├── src/                # Source code
│   ├── heuristic_solver.py
│   ├── entropy_solver.py
│   └── rl_solver.py    # Main RL training & evaluation script
├── notebooks/          # Exploratory data analysis (optional)
├── models/             # For storing pre-trained models
├── requirements.txt    # Pip dependencies
├── environment.yml     # Conda environment specification
└── README.md          # This file
```

## 🛠️ Dependencies

Core dependencies are listed in `requirements.txt` and `environment.yml`. The main libraries include:
- `torch` (PyTorch for deep learning)
- `numpy` (numerical computations)
- `matplotlib` (plotting results)
- `wordfreq` (for word frequency data)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Citation & Contact

If you use this code or find the project helpful for your research, please consider citing:

```bibtex
@software{YourName_WordleSolver_2024,
  author = {Your Name},
  title = {Advanced Wordle Solvers: Heuristic, Entropy, and Reinforcement Learning Approaches},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/your-repo-name}}
}
```

**Author**: [Your Name]  
**Email**: [your.email@manchester.ac.uk]  
**LinkedIn**: [Your LinkedIn Profile URL]  

---

*This project was completed as part of the MSc Data Science program at The University of Manchester.*

| Solver | Avg. Attempts (95% CI) | Success Rate | Key Innovation |
| :--- | :--- | :--- | :--- |
| **Heuristic** | ~3.85 | ~98.5% | Letter/position frequency scoring |
| **Entropy-Optimized** | **~3.62** | **~99.2%** | Second-guess lookup table |
| **RL (Ours)** | **~3.48** | **~99.6%** | **Dueling-DDQN + BC + CQL** |

*Results based on 3,000 simulation games per solver. The RL agent demonstrates statistically significant improvement over the baseline.*

## ⚙️ Repository Structure
