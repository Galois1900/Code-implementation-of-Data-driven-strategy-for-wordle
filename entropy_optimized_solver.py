# -*- coding: utf-8 -*-
"""
Wordle Solver with Information Entropy and Second-Guess Optimization
- Uses entropy-based scoring for optimal word selection
- Implements second-guess lookup table for improved performance
"""

import random
import math
import time
import csv
import os
from collections import Counter, defaultdict
from wordfreq import word_frequency
import builtins
import numpy as np
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

# ======================
# WORD FREQUENCY DATA
# ======================
WORD_FREQUENCY = {
    word: word_frequency(word, 'en') + 1e-9 for word in GUESS_WORDS
}

# ======================
# CORE GAME FUNCTIONS
# ======================
def get_feedback(guess, target):
    """Generate Wordle feedback pattern (G/Y/B)"""
    feedback = [''] * 5
    target_counts = Counter(target)
    
    # First pass: check for exact matches (Green)
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 'G'
            target_counts[guess[i]] -= 1
            
    # Second pass: check for partial matches (Yellow)
    for i in range(5):
        if feedback[i] == '':
            if guess[i] in target_counts and target_counts[guess[i]] > 0:
                feedback[i] = 'Y'
                target_counts[guess[i]] -= 1
            else:
                feedback[i] = 'B'
                
    return ''.join(feedback)

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

    # Apply filtering rules
    for word in words:
        match = True
        
        # Check green constraints (correct position)
        for i in range(5):
            if i in greens and word[i] != greens[i]:
                match = False
                break
            if (word[i] in blacks and 
                word[i] not in greens.values() and 
                word[i] not in yellows):
                match = False
                break
                
        # Check yellow constraints (wrong position but exists)
        for y_char, banned_pos in yellows.items():
            if y_char not in word:
                match = False
                break
            if any(word[pos] == y_char for pos in banned_pos):
                match = False
                break
                
        if match:
            filtered.append(word)
            
    return filtered

# ======================
# SCORING FUNCTIONS
# ======================
def compute_entropy_score(guess, candidates):
    """Calculate information entropy score for a guess"""
    feedback_counts = Counter()
    for target in candidates:
        feedback = get_feedback(guess, target)
        feedback_counts[feedback] += 1
        
    total = sum(feedback_counts.values())
    entropy = 0
    for count in feedback_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy

def compute_hit_score(guess, candidates):
    """Calculate hit score combining frequency and reduction potential"""
    freq = math.log(WORD_FREQUENCY.get(guess, 1e-9))
    total_reduction = 0
    
    for target in candidates:
        feedback = get_feedback(guess, target)
        reduced = len(candidates) - len(filter_words(candidates, guess, feedback))
        total_reduction += reduced
        
    avg_reduction = total_reduction / len(candidates)
    return freq + 0.02 * avg_reduction

def compute_letter_frequencies(words):
    """Calculate letter frequencies across all positions and individually"""
    letter_freq = Counter()
    position_freq = [Counter() for _ in range(5)]
    
    for word in words:
        seen = set()
        for i, c in enumerate(word):
            position_freq[i][c] += 1
            if c not in seen:
                letter_freq[c] += 1
                seen.add(c)
                
    return letter_freq, position_freq

def compute_heuristic_score(word, letter_freq, position_freq):
    """Calculate heuristic score based on letter and position frequencies"""
    score = 0
    used = set()
    
    for i, c in enumerate(word):
        if c not in used:
            score += letter_freq[c]
            used.add(c)
        score += position_freq[i][c]
        
    return score

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

def bootstrap_mean_ci(data, n_boot=5000, alpha=0.05, seed=42):
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
# SECOND-GUESS OPTIMIZATION
# ======================
def build_second_guess_lookup(first_guess="tarse"):
    """Build lookup table for optimal second guesses based on first feedback"""
    lookup = {}
    feedback_buckets = defaultdict(list)

    # Group targets by feedback from first guess
    for target in TARGET_WORDS:
        fb = get_feedback(first_guess, target)
        feedback_buckets[fb].append(target)

    # Find best second guess for each feedback pattern
    for fb, bucket in feedback_buckets.items():
        best_word = None
        best_score = -1
        
        for word in GUESS_WORDS:
            score = compute_entropy_score(word, bucket)
            if score > best_score:
                best_score = score
                best_word = word
                
        lookup[fb] = best_word
        
    return lookup

# Initialize second-guess lookup table
FIXED_FIRST = "tarse"
SECOND_GUESS_LOOKUP = build_second_guess_lookup(FIXED_FIRST)

# ======================
# MAIN SOLVER
# ======================
def play_wordle(target_word, return_trace=False):
    """Play a Wordle game with optimized guessing strategy"""
    candidates = TARGET_WORDS.copy()
    guessed = set()
    attempts = 0

    # For logging and analysis
    cand_sizes, feedbacks = [], []
    used_lookup = False

    # First fixed guess
    guess = FIXED_FIRST
    guessed.add(guess)
    feedback = get_feedback(guess, target_word)
    candidates = filter_words(candidates, guess, feedback)

    cand_sizes.append(len(candidates))
    feedbacks.append(feedback)

    if feedback == 'GGGGG':
        if return_trace:
            return 1, True, cand_sizes, feedbacks, used_lookup
        return 1, True
        
    attempts += 1

    # Subsequent guesses
    while attempts < 6:
        guess_scores = []
        letter_freq, position_freq = compute_letter_frequencies(candidates)

        # Use second-guess lookup if available
        if attempts == 1 and feedback in SECOND_GUESS_LOOKUP:
            guess = SECOND_GUESS_LOOKUP[feedback]
            used_lookup = True
            if guess in guessed:
                guess = random.choice([w for w in GUESS_WORDS if w not in guessed])
                
        # Use entropy scoring for small candidate sets
        elif len(candidates) <= 15:
            for word in candidates:
                if word in guessed: 
                    continue
                entropy = compute_entropy_score(word, candidates)
                guess_scores.append((word, entropy))
            guess = max(guess_scores, key=lambda x: x[1])[0]
            
        # Use hit scoring for larger candidate sets
        else:
            for word in candidates:
                if word in guessed: 
                    continue
                score = compute_hit_score(word, candidates)
                guess_scores.append((word, score))
            guess = max(guess_scores, key=lambda x: x[1])[0]

        guessed.add(guess)
        feedback = get_feedback(guess, target_word)
        candidates = filter_words(candidates, guess, feedback)

        cand_sizes.append(len(candidates))
        feedbacks.append(feedback)

        if feedback == 'GGGGG':
            if return_trace:
                return attempts+1, True, cand_sizes, feedbacks, used_lookup
            return attempts+1, True

        attempts += 1

    if return_trace:
        return 6, False, cand_sizes, feedbacks, used_lookup
    return 6, False

# ======================
# SIMULATION FUNCTIONS
# ======================
def simulate_games(n=3000):
    """Run multiple game simulations with output capture"""
    total_attempts = 0
    success_count = 0

    for _ in range(n):
        target = random.choice(ANSWERS)
        output_buffer = []

        # Capture print output
        def custom_print(*args, **kwargs):
            output_buffer.append(" ".join(str(arg) for arg in args))

        original_print = builtins.print
        builtins.print = custom_print
        
        try:
            attempts, success = play_wordle(target)
        finally:
            builtins.print = original_print

        total_attempts += attempts
        if success:
            success_count += 1
        else:
            print(f"\nTarget Word: {target.upper()}")
            for line in output_buffer:
                print(line)

    print("\n==== Final Result ====")
    print(f"Average attempts: {total_attempts / n:.2f}")
    print(f"Success rate: {success_count / n * 100:.2f}%")

def evaluate_with_logging(games=3000, seed=42, outdir="figs_eh_simple"):
    """Run simulations with detailed logging for analysis"""
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)

    attempts, success, runtimes = [], [], []
    round_sizes_list, feedbacks_list = [], []
    used_lookup_flags = []

    for _ in range(games):
        target = rng.choice(TARGET_WORDS)

        # Time the game
        t0 = time.perf_counter()
        att, ok, cand_sizes, fbs, used_lookup = play_wordle(target, return_trace=True)
        runtimes.append(time.perf_counter() - t0)

        attempts.append(att)
        success.append(1 if ok else 0)
        round_sizes_list.append(cand_sizes[:6])
        feedbacks_list.append(fbs[:6])
        used_lookup_flags.append(1 if used_lookup else 0)

    # Save results to CSV
    csv_path = os.path.join(outdir, "eh_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attempts", "success", "runtime_sec", "used_lookup", "round_sizes", "feedbacks"])
        for a, s, rt, lu, rs, fb in zip(attempts, success, runtimes, used_lookup_flags, round_sizes_list, feedbacks_list):
            w.writerow([a, s, f"{rt:.6f}", lu, "|".join(map(str, rs)), "|".join(fb)])
            
    print(f"[INFO] saved {csv_path}")
    return attempts, success, runtimes, round_sizes_list, feedbacks_list, used_lookup_flags

# ======================
# VISUALIZATION FUNCTIONS
# ======================
def plot_attempts_hist(attempts, path):
    """Plot histogram of attempts distribution"""
    bins = [1, 2, 3, 4, 5, 6]
    counts = [sum(1 for a in attempts if a == b) for b in bins]
    total = sum(counts)
    
    plt.figure(figsize=(6, 4))
    plt.bar([str(b) for b in bins], [c/total for c in counts], width=0.6)
    
    for x, c in zip(bins, counts):
        plt.text(x-1+0.3, c/total+0.01, f"{c/total*100:.1f}%", ha="center", fontsize=9)
        
    plt.xlabel("Final attempts")
    plt.ylabel("Proportion")
    plt.title("Entropy+Heuristic — Attempts distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _mean_ci95(x):
    """Calculate mean and 95% confidence interval"""
    x = np.asarray(x, float)
    if len(x) == 0: 
        return np.nan, 0.0
        
    m = np.mean(x)
    s = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return m, 1.96 * s / np.sqrt(max(1, len(x)))

def plot_round_sizes(round_sizes_list, path):
    """Plot candidate set size convergence over rounds"""
    # Aggregate each round
    buckets = [[] for _ in range(6)]
    for rs in round_sizes_list:
        for i, v in enumerate(rs[:6]):
            buckets[i].append(v)
            
    means, cis = zip(*(_mean_ci95(b) for b in buckets))
    x = np.arange(1, 7)
    
    plt.figure(figsize=(6.6, 4))
    plt.plot(x, means, marker="o")
    plt.fill_between(x, np.array(means)-np.array(cis), np.array(means)+np.array(cis), alpha=0.2)
    
    plt.xticks(x)
    plt.xlabel("Round")
    plt.ylabel("Candidate size (mean ± 95% CI)")
    plt.title("Entropy+Heuristic — Candidate set convergence")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_runtime_box(runtimes, path):
    """Plot runtime distribution as box plot"""
    data = np.asarray(runtimes, float)
    
    plt.figure(figsize=(5.2, 4))
    plt.boxplot([data], widths=0.5, patch_artist=True)
    plt.scatter([1], [data.mean()], marker="^", s=80, zorder=3, label="mean")
    
    plt.xticks([1], ["Entropy+Heuristic"])
    plt.ylabel("Runtime per game (sec)")
    plt.title("Runtime distribution")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_lookup_effect(used_lookup_flags, round_sizes_list, path):
    """Plot effect of second-guess lookup on performance"""
    flags = np.asarray(used_lookup_flags, int)

    used_2nd = []
    not_2nd = []
    
    for f, rs in zip(flags, round_sizes_list):
        if len(rs) >= 1:
            second = rs[0]  # Candidate size after first guess
            if f == 1: 
                used_2nd.append(second)
            else:    
                not_2nd.append(second)

    used_2nd = np.asarray(used_2nd, float)
    not_2nd = np.asarray(not_2nd, float)
    prop = flags.mean() * 100 if len(flags) else 0.0
    
    mu, ci = _mean_ci95(used_2nd) if len(used_2nd) else (np.nan, np.nan)
    mn, cin = _mean_ci95(not_2nd) if len(not_2nd) else (np.nan, np.nan)

    fig, ax1 = plt.subplots(figsize=(6.6, 4))
    ax2 = ax1.twinx()

    ax1.bar(["Lookup used"], [prop], width=0.6, alpha=0.9)
    ax1.set_ylabel("2nd-guess lookup usage (%)")

    ax2.errorbar([0.2, 0.8], [mu, mn], yerr=[ci, cin], fmt="o-", capsize=4)
    ax2.set_ylabel("Candidate size at Round-2 (mean ± 95% CI)")

    plt.title("Effect of 2nd-guess lookup")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

# ======================
# MAIN ANALYSIS FUNCTION
# ======================
def run_plots_simple(games=3000, outdir="figs_eh_simple"):
    """Run complete analysis with visualization"""
    atts, succ, rts, rs_list, fb_list, lu_flags = evaluate_with_logging(games, outdir)
    n = len(atts)
    
    # Calculate statistics
    avg_att = float(np.mean(atts))
    att_lo, att_hi = bootstrap_mean_ci(atts, n_boot=3000, alpha=0.05, seed=2025)

    succ_cnt = int(np.sum(succ))
    succ_rate = succ_cnt / n
    sr_lo, sr_hi = wilson_ci(succ_cnt, n, z=1.96)

    rt_mean = float(np.mean(rts))
    rt_median = float(np.median(rts))
    rt_p90 = float(np.percentile(rts, 90))

    # Print summary
    print(f"[Summary][EH] games={n}")
    print(f"[Summary][EH] avg_attempts={avg_att:.2f} (95% CI: {att_lo:.2f}–{att_hi:.2f}) | "
          f"success_rate={succ_rate*100:.2f}% (95% CI: {sr_lo*100:.2f}–{sr_hi*100:.2f}%)")
    print(f"[Summary][EH] runtime: mean={rt_mean:.4f}s, median={rt_median:.4f}s, P90={rt_p90:.4f}s")
    print("[Protocol] Average attempts include failures recorded as six guesses; "
          "Wilson CI for proportions, bootstrap CI for means.")

    # Generate plots
    plot_attempts_hist(atts, os.path.join(outdir, "fig_attempts_hist.png"))
    plot_round_sizes(rs_list, os.path.join(outdir, "fig_round_candidates.png"))
    plot_runtime_box(rts, os.path.join(outdir, "fig_runtime_distribution.png"))
    plot_lookup_effect(lu_flags, rs_list, os.path.join(outdir, "fig_lookup_effect.png"))

    print(f"[DONE] saved to ./{outdir}")

# ======================
# EXECUTION
# ======================
if __name__ == "__main__":
    # Run simulation with visualization
    run_plots_simple(games=5000, outdir="figs_eh_simple")