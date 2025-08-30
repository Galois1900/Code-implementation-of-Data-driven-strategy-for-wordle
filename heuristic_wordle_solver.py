# -*- coding: utf-8 -*-
"""
Heuristic Wordle Solver with Visualization
- Pure heuristic approach (no RL)
- Generates publication-ready figures with consistent styling
"""

import random
import time
import os
import math
from collections import Counter, defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ======================
# DATA LOADING
# ======================
with open("wordlist_hidden.txt", "r") as f:
    ANSWERS = [line.strip().lower() for line in f if len(line.strip()) == 5]
with open("wordlist_all.txt", "r") as f:
    ALLOWED = [line.strip().lower() for line in f if len(line.strip()) == 5]

GUESS_WORDS = sorted(set(ANSWERS + ALLOWED))
TARGET_WORDS = ANSWERS.copy()
FIRST_GUESS = "tarse"

# ======================
# HEURISTIC SOLVER CORE
# ======================
def compute_letter_frequencies(words):
    """Calculate letter frequencies across all positions and individually"""
    letter_freq = Counter()
    position_freq = [Counter() for _ in range(5)]
    
    for w in words:
        seen = set(w)
        letter_freq.update(seen)
        for i, c in enumerate(w):
            position_freq[i][c] += 1
            
    return letter_freq, position_freq

def score_word(word, letter_freq, position_freq):
    """Score word based on letter and position frequencies"""
    s, used = 0, set()
    for i, c in enumerate(word):
        if c not in used:
            s += letter_freq[c] * 1
            used.add(c)
        s += position_freq[i][c]
    return s

def get_feedback(guess, target):
    """Generate Wordle feedback (G/Y/B) for a guess against target"""
    fb = [''] * 5
    cnt = Counter(target)
    
    # First pass: check for exact matches (Green)
    for i in range(5):
        if guess[i] == target[i]:
            fb[i] = 'G'
            cnt[guess[i]] -= 1
            
    # Second pass: check for partial matches (Yellow)
    for i in range(5):
        if fb[i] == '':
            if cnt.get(guess[i], 0) > 0:
                fb[i] = 'Y'
                cnt[guess[i]] -= 1
            else:
                fb[i] = 'B'
                
    return ''.join(fb)

def filter_words(words, guess, feedback):
    """Filter word list based on guess feedback"""
    filtered = []
    greens = {i: guess[i] for i in range(5) if feedback[i] == 'G'}
    yellows = defaultdict(set)
    blacks = set()
    
    # Process feedback
    for i in range(5):
        if feedback[i] == 'Y': 
            yellows[guess[i]].add(i)
        elif feedback[i] == 'B': 
            blacks.add(guess[i])

    # Filter words based on feedback
    for w in words:
        ok = True
        
        # Check green constraints
        for i in range(5):
            if i in greens and w[i] != greens[i]: 
                ok = False
                break
            if (w[i] in blacks and 
                w[i] not in greens.values() and 
                w[i] not in yellows): 
                ok = False
                break
                
        # Check yellow constraints
        if ok:
            for y, ban in yellows.items():
                if y not in w: 
                    ok = False
                    break
                if any(w[pos] == y for pos in ban): 
                    ok = False
                    break
                    
        if ok: 
            filtered.append(w)
            
    return filtered

# ======================
# STATISTICAL FUNCTIONS
# ======================
def wilson_ci(successes, n, z=1.96):
    """Calculate 95% Wilson confidence interval for a proportion"""
    if n == 0:
        return (float("nan"), float("nan"))
        
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    margin = (z / denom) * math.sqrt((phat*(1 - phat) / n) + (z**2) / (4 * n**2))
    
    return max(0.0, center - margin), min(1.0, center + margin)

def bootstrap_mean_ci(data, n_boot=2000, alpha=0.05, seed=42):
    """Calculate nonparametric bootstrap 95% CI for the mean"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, float)
    
    if arr.size == 0:
        return (float("nan"), float("nan"))
        
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1 - alpha/2))
    
    return lo, hi

# ======================
# GAME SIMULATION
# ======================
def play_wordle_logged(target_word):
    """Play a single Wordle game with detailed logging"""
    t0 = time.time()
    candidates = TARGET_WORDS.copy()
    guessed = set()
    attempts = 0
    cand_sizes, feedbacks = [], []
    guesses = []  # Track guesses made

    # First guess
    guess = FIRST_GUESS
    guessed.add(guess)
    guesses.append(guess)
    fb = get_feedback(guess, target_word)
    candidates = filter_words(candidates, guess, fb)
    cand_sizes.append(len(candidates))
    feedbacks.append(fb)
    
    if fb == 'GGGGG':
        return attempts+1, True, cand_sizes, feedbacks, guesses, time.time()-t0
        
    attempts += 1

    # Subsequent guesses
    USE_G_FOR_GUESS = True  # Whether to use full guess word list
    
    while attempts < 6:
        letter_freq, position_freq = compute_letter_frequencies(candidates)
        pool_words = GUESS_WORDS if USE_G_FOR_GUESS else candidates
        pool = [(w, score_word(w, letter_freq, position_freq)) 
                for w in pool_words if w not in guessed]
                
        if not pool:
            return 6, False, cand_sizes, feedbacks, guesses, time.time()-t0
            
        guess = max(pool, key=lambda x: x[1])[0]
        guesses.append(guess)
        guessed.add(guess)
        fb = get_feedback(guess, target_word)
        candidates = filter_words(candidates, guess, fb)
        cand_sizes.append(len(candidates))
        feedbacks.append(fb)
        
        if fb == 'GGGGG':
            return attempts+1, True, cand_sizes, feedbacks, guesses, time.time()-t0
            
        attempts += 1

    return 6, False, cand_sizes, feedbacks, guesses, time.time()-t0

def simulate_and_log(n=3000, seed=42):
    """Run multiple game simulations and collect results"""
    rng = random.Random(seed)
    attempts, success, runtimes = [], [], []
    all_sizes, all_fbs, all_guesses, all_targets = [], [], [], []
    
    for _ in range(n):
        tgt = rng.choice(ANSWERS)
        att, ok, sizes, fbs, guesses, dt = play_wordle_logged(tgt)
        
        attempts.append(att)
        success.append(int(ok))
        runtimes.append(dt)
        all_sizes.append(sizes)
        all_fbs.append(fbs)
        all_guesses.append(guesses)
        all_targets.append(tgt)
        
    return attempts, success, runtimes, all_sizes, all_fbs, all_guesses, all_targets

# ======================
# VISUALIZATION SETTINGS
# ======================
def _set_mpl_style():
    """Set matplotlib style for consistent plotting"""
    mpl.rcParams.update({
        "figure.figsize": (6.8, 4.5),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
    })

_TAB = plt.get_cmap("tab10").colors
BLUE, ORANGE = _TAB[0], _TAB[1]

def _ensure_dir(d): 
    """Ensure directory exists"""
    if d and not os.path.exists(d): 
        os.makedirs(d, exist_ok=True)

def _mean_ci_95(arr):
    """Calculate mean and 95% confidence interval"""
    arr = np.asarray(arr, float)
    if arr.size == 0: 
        return np.nan, 0.0
        
    m = float(np.nanmean(arr))
    if arr.size < 2: 
        return m, 0.0
        
    s = float(np.nanstd(arr, ddof=1))
    ci = 1.96 * s / math.sqrt(arr.size)
    
    return m, ci

# ======================
# PLOTTING FUNCTIONS
# ======================
def plot_round_candidates(all_cand_sizes, savepath):
    """Plot candidate set size convergence over rounds"""
    _set_mpl_style()
    _ensure_dir(os.path.dirname(savepath))
    
    buckets = [[] for _ in range(6)]
    for sizes in all_cand_sizes:
        for i, v in enumerate(sizes[:6]): 
            buckets[i].append(v)
            
    means, cis = zip(*(_mean_ci_95(b) for b in buckets))
    x = np.arange(1, 7)
    
    fig, ax = plt.subplots()
    ax.plot(x, means, color=BLUE, linewidth=2.0, marker="o", markersize=6)
    ax.fill_between(x, np.array(means)-np.array(cis), np.array(means)+np.array(cis), 
                    color=BLUE, alpha=0.18)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Candidate set size (mean ± 95% CI)")
    ax.set_title("Heuristic Baseline — Candidate Set Convergence")
    ax.set_xticks(x)
    ax.margins(x=0.02)
    
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

def plot_runtime_box(runtimes, savepath):
    """Plot runtime distribution as violin + box plot"""
    _set_mpl_style()
    _ensure_dir(os.path.dirname(savepath))
    
    data = np.asarray(runtimes, float)
    
    fig, ax = plt.subplots()
    parts = ax.violinplot([data], showextrema=False, widths=0.9)
    
    for pc in parts['bodies']:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.18)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)
        
    ax.boxplot([data], widths=0.25, patch_artist=True,
               boxprops=dict(facecolor="#fff", edgecolor=BLUE, linewidth=1.6),
               medianprops=dict(color="red", linewidth=2),
               whiskerprops=dict(color=BLUE), 
               capprops=dict(color=BLUE))
               
    ax.scatter([1], [data.mean()], marker="^", s=70, color=ORANGE, 
               zorder=3, label="mean")
               
    ax.set_xticks([1])
    ax.set_xticklabels(["Heuristic baseline"])
    ax.set_ylabel("Runtime per game (seconds)")
    ax.set_title("Heuristic Baseline — Runtime Distribution")
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

def plot_attempt_hist(attempts_list, savepath, normalize=False):
    """Plot attempts distribution histogram"""
    _set_mpl_style()
    _ensure_dir(os.path.dirname(savepath))
    
    bins = [1, 2, 3, 4, 5, 6]
    counts = np.array([sum(1 for a in attempts_list if a == b) for b in bins], int)
    total = counts.sum()
    heights = counts / total if normalize else counts
    
    fig, ax = plt.subplots()
    bars = ax.bar([str(b) for b in bins], heights, width=0.62, color=BLUE, alpha=0.85)
    
    for rect, c in zip(bars, counts):
        ratio = (c / total * 100.0) if total > 0 else 0.0
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + (heights.max() * 0.02 if heights.max() > 0 else 0.02),
                f"{c} ({ratio:.1f}%)", ha="center", va="bottom", fontsize=10)
                
    ax.set_xlabel("Attempts (final)")
    ax.set_ylabel("Proportion" if normalize else "Number of games")
    ax.set_title("Heuristic Baseline — Attempts Distribution")
    
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

def plot_single_trajectory(all_cand_sizes, all_feedbacks, attempts_list, success_list,
                           all_guesses, all_targets, savepath, txt_outpath=None,
                           index=None, target=None, prefer_unsolved=True):
    """Plot candidate set trajectory for a single game"""
    _set_mpl_style()
    _ensure_dir(os.path.dirname(savepath))

    # Select game index
    idx = None
    if index is not None:
        idx = int(index)
    elif target is not None:
        for i, t in enumerate(all_targets):
            if t == target.lower():
                idx = i
                break
        if idx is None:
            print(f"[WARN] target '{target}' not found in this simulation batch.")
    if idx is None and prefer_unsolved:
        idx = next((i for i, ok in enumerate(success_list) if ok == 0), None)
    if idx is None:
        idx = int(np.argmax(attempts_list))

    # Prepare data
    sizes = np.asarray(all_cand_sizes[idx], float)
    fbs = all_feedbacks[idx]
    gseq = all_guesses[idx] if idx < len(all_guesses) else []
    tgt = all_targets[idx] if idx < len(all_targets) else "unknown"
    rounds = np.arange(1, len(sizes) + 1)

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(rounds, sizes, color=BLUE, linewidth=2.0, marker="o", markersize=6)
    
    for r, s, fb in zip(rounds, sizes, fbs):
        ax.annotate(fb, (r, s), textcoords="offset points", xytext=(0, 8),
                    ha="center", va="bottom", fontsize=10)
                    
    ax.set_xlabel("Round")
    ax.set_ylabel("Candidate set size")
    ax.set_title("Heuristic Baseline — Single-Game Trajectory (Difficult Case)")
    ax.set_xticks(rounds)
    ax.margins(x=0.03)
    
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

    # Output game details to text file
    lines = [f"Target word: {tgt}  |  Result: {'SUCCESS' if success_list[idx] == 1 else 'UNSOLVED'}"]
    for r, (g, fb) in enumerate(zip(gseq, fbs), start=1):
        lines.append(f"Round {r}: guess='{g}', feedback={fb}")
        
    if txt_outpath:
        _ensure_dir(os.path.dirname(txt_outpath))
        with open(txt_outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    print("[Difficult case] Guess & feedback sequence:")
    for ln in lines:
        print("  " + ln)

def plot_word_scores(words, letter_freq, position_freq, savepath="fig_word_scores.png"):
    """Plot comparison of word scores"""
    _set_mpl_style()
    scores = [score_word(word, letter_freq, position_freq) for word in words]
    
    fig, ax = plt.subplots()
    bars = ax.bar(words, scores, color=BLUE, alpha=0.7)
    
    ax.set_xlabel("Words")
    ax.set_ylabel("Score")
    ax.set_title("Heuristic Baseline — Word Scores Comparison")
    plt.xticks(rotation=45)
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{score}', ha='center', va='bottom')
                
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

# ======================
# MAIN EXECUTION
# ======================
def run_all_plots(n=3000, seed=42, outdir="figs_baseline"):
    """Run all simulations and generate plots"""
    _set_mpl_style()
    _ensure_dir(outdir)
    
    attempts, success, runtimes, all_sizes, all_fbs, all_guesses, all_targets = simulate_and_log(n=n, seed=seed)

    print(f"[Summary] games={n}, avg_attempts={np.mean(attempts):.2f}, success_rate={np.mean(success)*100:.2f}%")
    print(f"[Summary] runtime: mean={np.mean(runtimes):.4f}s, median={np.median(runtimes):.4f}s")

    plot_attempt_hist(attempts, os.path.join(outdir, "fig_attempts_hist.png"), normalize=False)
    plot_round_candidates(all_sizes, os.path.join(outdir, "fig_round_candidates.png"))
    plot_runtime_box(runtimes, os.path.join(outdir, "fig_runtime_distribution.png"))

    # Plot a difficult case trajectory
    plot_single_trajectory(all_sizes, all_fbs, attempts, success, all_guesses, all_targets,
                           os.path.join(outdir, "fig_single_trajectory.png"),
                           txt_outpath=os.path.join(outdir, "fig_single_trajectory_log.txt"),
                           prefer_unsolved=True)

if __name__ == "__main__":
    run_all_plots(n=3000, seed=42)