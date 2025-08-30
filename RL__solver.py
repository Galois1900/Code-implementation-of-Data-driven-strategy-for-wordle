# -*- coding: utf-8 -*-
"""
Wordle Offline Reinforcement Learning (Enhanced Version):
- Demo data: 30,000 games; oversampling difficult words; Top-K entropy splitting words added from step 3
- Model: Dueling Double DQN + action masking + behavioral cloning (BC) + Conservative Q (CQL)
- Training: SmoothL1, y-clipping [-15,5], cosine annealing, gradient clipping, target network soft update
- Strategy: Evaluation with epsilon=0, forced lookup in step 2; action set = candidates ∪ (lookup words) ∪ (splitting words when candidate pool is large)
"""

import os
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================
# Configuration & Random Seeds
# =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Vocabulary Loading (with fallback)
# =====================
ABS_ANSWERS = "wordlist_hidden.txt"
ABS_ALLOWED = "wordlist_all.txt"
REL_ANSWERS = "wordlist_hidden.txt"
REL_ALLOWED = "wordlist_all.txt"

ANSWERS_PATH = ABS_ANSWERS if os.path.exists(ABS_ANSWERS) else REL_ANSWERS
ALLOWED_PATH = ABS_ALLOWED if os.path.exists(ABS_ALLOWED) else REL_ALLOWED

with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    ANSWERS = [line.strip().lower() for line in f if len(line.strip()) == 5]
with open(ALLOWED_PATH, "r", encoding="utf-8") as f:
    ALLOWED = [line.strip().lower() for line in f if len(line.strip()) == 5]

GUESS_WORDS = sorted(set(ANSWERS + ALLOWED))
TARGET_WORDS = ANSWERS.copy()
WORD2IDX = {w: i for i, w in enumerate(GUESS_WORDS)}
IDX2WORD = {i: w for w, i in WORD2IDX.items()}
VOCAB_SIZE = len(GUESS_WORDS)

# Word frequency (optional)
try:
    from wordfreq import word_frequency as _wf
    WORD_FREQUENCY = {w: _wf(w, 'en') + 1e-9 for w in GUESS_WORDS}
except Exception:
    print("[WARN] wordfreq not available, using uniform word frequency.")
    WORD_FREQUENCY = {w: 1.0 for w in GUESS_WORDS}

# =====================
# Wordle Rules & Scoring
# =====================
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
FIXED_FIRST = 'tarse'
HARD_WORDS = {"wowed", "mousy", "shine", "filed", "queue", "jazzy", "fuzzy"}

def get_feedback(guess: str, target: str) -> str:
    """Generate Wordle feedback pattern (G/Y/B)"""
    fb = [''] * 5
    cnt = Counter(target)
    for i in range(5):
        if guess[i] == target[i]:
            fb[i] = 'G'
            cnt[guess[i]] -= 1
    for i in range(5):
        if fb[i] == '':
            c = guess[i]
            if cnt.get(c, 0) > 0:
                fb[i] = 'Y'
                cnt[c] -= 1
            else:
                fb[i] = 'B'
    return ''.join(fb)

def compute_entropy_score(guess: str, candidates: List[str]) -> float:
    """Calculate information entropy score for a guess"""
    counts = Counter(get_feedback(guess, t) for t in candidates)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / max(1, total)
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def compute_hit_score(guess: str, candidates: List[str]) -> float:
    """Calculate hit score combining frequency and reduction potential"""
    freq = math.log(WORD_FREQUENCY.get(guess, 1e-9))
    total_red = 0
    for t in candidates:
        fb = get_feedback(guess, t)
        red = len(candidates) - len(filter_words(candidates, guess, fb))
        total_red += red
    return freq + 0.02 * (total_red / max(1, len(candidates)))

def compute_letter_frequencies(words: List[str]):
    """Calculate letter frequencies across all positions and individually"""
    letter_freq = Counter()
    pos_freq = [Counter() for _ in range(5)]
    for w in words:
        seen = set()
        for i, c in enumerate(w):
            pos_freq[i][c] += 1
            if c not in seen:
                letter_freq[c] += 1
                seen.add(c)
    return letter_freq, pos_freq

def filter_words(words: List[str], guess: str, feedback: str) -> List[str]:
    """Filter word list based on guess feedback"""
    filtered = []
    greens = {i: guess[i] for i in range(5) if feedback[i] == 'G'}
    yellows = defaultdict(set)
    blacks = set()
    for i in range(5):
        if feedback[i] == 'Y':
            yellows[guess[i]].add(i)
        elif feedback[i] == 'B':
            blacks.add(guess[i])
    for w in words:
        ok = True
        for i in range(5):
            if i in greens and w[i] != greens[i]:
                ok = False; break
            if w[i] in blacks and w[i] not in greens.values() and w[i] not in yellows:
                ok = False; break
        if ok:
            for y, ban in yellows.items():
                if y not in w:
                    ok = False; break
                if any(w[p] == y for p in ban):
                    ok = False; break
        if ok:
            filtered.append(w)
    return filtered

# =====================
# Second Guess Lookup & Baseline
# =====================
def build_second_guess_lookup(first_guess: str = FIXED_FIRST) -> Dict[str, str]:
    """Build lookup table for optimal second guesses based on first feedback"""
    lookup = {}
    buckets = defaultdict(list)
    for tgt in TARGET_WORDS:
        buckets[get_feedback(first_guess, tgt)].append(tgt)
    for fb, bkt in buckets.items():
        best_word = None; best = -1
        for w in GUESS_WORDS:
            s = compute_entropy_score(w, bkt)
            if s > best:
                best, best_word = s, w
        lookup[fb] = best_word
    return lookup

SECOND_GUESS_LOOKUP = build_second_guess_lookup(FIXED_FIRST)

# =====================
# Original Baseline Solver (for simulation and comparison)
# =====================
def play_wordle_original(target_word: str, return_trace: bool = False):
    """Original baseline solver for Wordle"""
    candidates = TARGET_WORDS.copy()
    guessed = set(); attempts = 0
    cand_sizes = []; feedbacks = []; used_lookup = False

    guess = FIXED_FIRST
    fb = get_feedback(guess, target_word)
    candidates = filter_words(candidates, guess, fb)
    cand_sizes.append(len(candidates)); feedbacks.append(fb)
    if fb == 'GGGGG':
        return (1, True, cand_sizes, feedbacks, used_lookup) if return_trace else (1, True)
    guessed.add(guess); attempts += 1

    while attempts < 6:
        scores = []
        if attempts == 1 and fb in SECOND_GUESS_LOOKUP:
            guess = SECOND_GUESS_LOOKUP[fb]; used_lookup = True
            if guess in guessed:
                guess = random.choice([w for w in GUESS_WORDS if w not in guessed])
        elif len(candidates) <= 15:
            for w in candidates:
                if w in guessed: continue
                scores.append((w, compute_entropy_score(w, candidates)))
            guess = max(scores, key=lambda x: x[1])[0] if scores else random.choice(candidates)
        else:
            for w in candidates:
                if w in guessed: continue
                scores.append((w, compute_hit_score(w, candidates)))
            guess = max(scores, key=lambda x: x[1])[0] if scores else random.choice(candidates)

        fb = get_feedback(guess, target_word)
        candidates = filter_words(candidates, guess, fb)
        cand_sizes.append(len(candidates)); feedbacks.append(fb)
        if fb == 'GGGGG':
            return (attempts + 1, True, cand_sizes, feedbacks, used_lookup) if return_trace else (attempts + 1, True)
        guessed.add(guess); attempts += 1

    return (6, False, cand_sizes, feedbacks, used_lookup) if return_trace else (6, False)

# =====================
# State Features & Allowed Actions
# =====================
def extract_state_features(candidates: List[str], round_number: int) -> np.ndarray:
    """Extract state features: 1 + (5*26) + 26 = 157 dimensions"""
    n = len(candidates)
    feats = [math.log(n + 1)]
    letter_freq, pos_freq = compute_letter_frequencies(candidates)
    for i in range(5):
        for c in ALPHABET:
            feats.append(pos_freq[i].get(c, 0) / (n + 1e-9))
    for c in ALPHABET:
        feats.append(letter_freq.get(c, 0) / (n + 1e-9))
    return np.array(feats, dtype=np.float32)

SPLIT_SAMPLE_POOL = 2000  # Pool size for calculating splitting words (for speed)

def dyn_split_k(n: int) -> int:
    """Adaptive Top-K: more splitting words for larger candidate sets; limited to [100, 800]"""
    return max(100, min(800, int(30 * math.sqrt(max(1, n)))))

def select_splitters(candidates: List[str], guessed: set, k: Optional[int] = None, pool: int = SPLIT_SAMPLE_POOL) -> List[str]:
    """Select top-K splitting words based on entropy"""
    if len(candidates) <= 1:
        return []
    if k is None:
        k = dyn_split_k(len(candidates))
    pool_words = [w for w in GUESS_WORDS if w not in guessed]
    if len(pool_words) > pool:
        pool_words = random.sample(pool_words, pool)
    scored = []
    for w in pool_words:
        s = compute_entropy_score(w, candidates)
        scored.append((s, w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:k]]

def allowed_action_words(candidates: List[str], guessed: set, attempts: int, last_fb: str, diversify: bool = False) -> List[str]:
    """Get allowed action words for current state"""
    allow = [w for w in candidates if w not in guessed]
    # Step 2: Force lookup
    if attempts == 1 and last_fb in SECOND_GUESS_LOOKUP:
        g2 = SECOND_GUESS_LOOKUP[last_fb]
        if g2 not in guessed and g2 not in allow:
            allow.append(g2)
    # Add Top-K splitting words when candidate pool is large and diversification is needed
    if diversify and len(candidates) > 15 and attempts >= 2:
        for w in select_splitters(candidates, guessed, None, SPLIT_SAMPLE_POOL):
            if w not in guessed:
                allow.append(w)
    return list(dict.fromkeys(allow))

# =====================
# Demo Data Collection (30k games; hard word oversampling & diversification)
# =====================
def collect_training_data(num_games: int = 30000, seed: int = 42, hard_ratio: float = 0.2, diversify_prob: float = 0.7):
    """Collect training data with hard word oversampling and diversification"""
    rng = random.Random(seed)
    S, A, R, S2, D = [], [], [], [], []
    allow_s, allow_s2 = [], []

    print(f"Starting collection of {num_games} game data...")
    t0 = time.perf_counter()

    for g in range(num_games):
        # Oversample hard words
        if rng.random() < hard_ratio:
            target = rng.choice(list(HARD_WORDS))
        else:
            target = rng.choice(TARGET_WORDS)

        candidates = TARGET_WORDS.copy()
        guessed = set(); attempts = 0
        diversify = (rng.random() < diversify_prob)

        # First fixed guess
        guess = FIXED_FIRST
        fb = get_feedback(guess, target)
        next_candidates = filter_words(candidates, guess, fb)

        s = extract_state_features(candidates, attempts)
        s2 = extract_state_features(next_candidates, attempts + 1)
        allow_curr = [WORD2IDX[FIXED_FIRST]]
        allow_next = [WORD2IDX[w] for w in allowed_action_words(next_candidates, guessed | {guess}, attempts + 1, fb, diversify)]

        S.append(s); A.append(WORD2IDX[guess]); R.append(-1.0); S2.append(s2); D.append(float(fb == 'GGGGG'))
        # Hit reward (rare for first step, but consistent)
        if fb == 'GGGGG':
            R[-1] += 5.0
        allow_s.append(allow_curr); allow_s2.append(allow_next)

        candidates = next_candidates; guessed.add(guess); attempts += 1; last_fb = fb
        done = (fb == 'GGGGG')

        while attempts < 6 and not done:
            s = extract_state_features(candidates, attempts)
            allow_curr = [WORD2IDX[w] for w in allowed_action_words(candidates, guessed, attempts, last_fb, diversify)]

            # Demo strategy: same as baseline + diversified samples
            if attempts == 1 and last_fb in SECOND_GUESS_LOOKUP:
                guess = SECOND_GUESS_LOOKUP[last_fb]
                if guess in guessed:
                    guess = rng.choice([w for w in GUESS_WORDS if w not in guessed])
            elif len(candidates) <= 15:
                scored = [(w, compute_entropy_score(w, candidates)) for w in candidates if w not in guessed]
                guess = max(scored, key=lambda x: x[1])[0] if scored else rng.choice(candidates)
            else:
                # If diversify=True, allow selection from allow_curr (includes splitting words) based on entropy
                pool = [w for w in (allow_curr and [IDX2WORD[i] for i in allow_curr] or candidates) if w not in guessed]
                scored = [(w, compute_entropy_score(w, candidates)) for w in pool]
                guess = max(scored, key=lambda x: x[1])[0] if scored else rng.choice(pool)

            fb = get_feedback(guess, target)
            next_candidates = filter_words(candidates, guess, fb)

            s2 = extract_state_features(next_candidates, attempts + 1)
            allow_next = [WORD2IDX[w] for w in allowed_action_words(next_candidates, guessed | {guess}, attempts + 1, fb, diversify)]

            S.append(s); A.append(WORD2IDX[guess]); R.append(-1.0); S2.append(s2); D.append(float(fb == 'GGGGG'))
            if fb == 'GGGGG':
                R[-1] += 5.0  # Success reward, encourage faster hits
            allow_s.append(allow_curr); allow_s2.append(allow_next)

            candidates = next_candidates; guessed.add(guess); attempts += 1; last_fb = fb
            done = (fb == 'GGGGG')

        # Additional penalty for failure (applied to last transition of the game)
        if not done:
            R[-1] += -10.0

        if (g + 1) % 1000 == 0:
            print(f"Completed {g + 1}/{num_games}, time elapsed {time.perf_counter() - t0:.1f}s")

    S = np.asarray(S, np.float32); A = np.asarray(A, np.int64); R = np.asarray(R, np.float32)
    S2 = np.asarray(S2, np.float32); D = np.asarray(D, np.float32)
    print(f"Data collection completed: {len(S)} samples")
    return S, A, R, S2, D, allow_s, allow_s2

# =====================
# Dataset & Collate (variable-length allowed lists)
# =====================
class WordleDataset(Dataset):
    """Dataset for Wordle training data"""
    def __init__(self, S, A, R, S2, D, allow_s, allow_s2):
        self.S = torch.tensor(S, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.long)
        self.R = torch.tensor(R, dtype=torch.float32)
        self.S2 = torch.tensor(S2, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.allow_s = allow_s
        self.allow_s2 = allow_s2
    def __len__(self): return len(self.A)
    def __getitem__(self, i):
        return (self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i], self.allow_s[i], self.allow_s2[i])

def collate_varlen(batch):
    """Collate function for variable-length allowed lists"""
    S, A, R, S2, D, allow_s, allow_s2 = zip(*batch)
    S = torch.stack(S, dim=0)
    A = torch.stack(A, dim=0)
    R = torch.stack(R, dim=0)
    S2 = torch.stack(S2, dim=0)
    D = torch.stack(D, dim=0)
    return S, A, R, S2, D, list(allow_s), list(allow_s2)

# =====================
# Dueling Q Network
# =====================
class QNetwork(nn.Module):
    """Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)"""
    def __init__(self, input_size: int, output_size: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.V = nn.Linear(hidden, 1)
        self.A = nn.Linear(hidden, output_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        v = self.V(x)              # [B,1]
        a = self.A(x)              # [B,|A|]
        return v + a - a.mean(dim=1, keepdim=True)

# =====================
# Training (Double DQN + masking + BC + CQL + soft update)
# =====================
@dataclass
class TrainCfg:
    """Training configuration"""
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    grad_clip: float = 0.5
    bc_neg_k: int = 16
    bc_margin: float = 0.3
    bc_weight: float = 0.15
    cql_weight: float = 0.05
    y_clip_min: float = -15.0
    y_clip_max: float = 5.0
    tau: float = 0.01   # Target network soft update coefficient

def train_double_dqn_masked(model: QNetwork, dataset: WordleDataset, cfg: TrainCfg = TrainCfg()):
    """Train Double DQN with masking, behavioral cloning, and conservative Q-learning"""
    model.to(device)
    target = QNetwork(model.fc1.in_features, model.A.out_features).to(device)
    target.load_state_dict(model.state_dict())

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=3e-5)
    huber = nn.SmoothL1Loss()

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=collate_varlen)

    for ep in range(cfg.epochs):
        ep_loss = 0.0; t0 = time.perf_counter()
        for S, A, R, S2, D, allow_s, allow_s2 in loader:
            S = S.to(device); A = A.to(device); R = R.to(device); S2 = S2.to(device); D = D.to(device)

            # Current Q(s,a)
            q_all = model(S)
            q_sa = q_all.gather(1, A.unsqueeze(1)).squeeze(1)

            # Double DQN target (select a* on allow_s2)
            with torch.no_grad():
                q2_online = model(S2)
                a_star_list = []
                for b, allow in enumerate(allow_s2):
                    if not allow:
                        a_star_list.append(0)
                    else:
                        vals = q2_online[b, allow]
                        a_star_list.append(int(allow[int(torch.argmax(vals).item())]))
                a_star = torch.tensor(a_star_list, dtype=torch.long, device=device)

                q2_target = target(S2)
                q2_star = q2_target.gather(1, a_star.unsqueeze(1)).squeeze(1)
                y = R + cfg.gamma * (1.0 - D) * q2_star
                y = torch.clamp(y, min=cfg.y_clip_min, max=cfg.y_clip_max)

            dqn_loss = huber(q_sa, y)

            # Behavioral cloning (margin ranking): make behavioral action > negative samples + margin
            bc_losses = []
            for j, allow in enumerate(allow_s):
                neg_pool = [idx for idx in allow if idx != int(A[j])]
                if not neg_pool:
                    continue
                if len(neg_pool) > cfg.bc_neg_k:
                    neg_idx = random.sample(neg_pool, cfg.bc_neg_k)
                else:
                    neg_idx = neg_pool
                idx_tensor = torch.tensor(neg_idx, dtype=torch.long, device=device)
                q_neg = q_all[j, idx_tensor]
                margin_term = cfg.bc_margin - (q_sa[j] - torch.max(q_neg))
                bc_losses.append(torch.clamp(margin_term, min=0.0))
            bc_loss = torch.stack(bc_losses).mean() if bc_losses else torch.tensor(0.0, device=device)

            # Conservative Q: suppress overestimation logsumexp(Q(s, allow)) - Q(s, a_beh)
            cql_terms = []
            for j, allow in enumerate(allow_s):
                if not allow:
                    continue
                idx_tensor = torch.tensor(allow, dtype=torch.long, device=device)
                lse = torch.logsumexp(q_all[j, idx_tensor], dim=0)
                cql_terms.append(lse - q_sa[j])
            cql_loss = torch.stack(cql_terms).mean() if cql_terms else torch.tensor(0.0, device=device)

            loss = dqn_loss + cfg.bc_weight * bc_loss + cfg.cql_weight * cql_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # Target network soft update
            with torch.no_grad():
                for p_t, p in zip(target.parameters(), model.parameters()):
                    p_t.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            ep_loss += loss.item() * S.size(0)

        scheduler.step()
        print(f"[TRAIN] epoch {ep+1}/{cfg.epochs}  loss={ep_loss/len(dataset):.6f}  lr={scheduler.get_last_lr()[0]:.2e}  time={time.perf_counter()-t0:.1f}s")

    return model

# =====================
# Inference & Evaluation (epsilon=0, forced lookup in step 2)
# =====================
def pick_allowed_words_eval(candidates: List[str], guessed: set, attempts: int, last_fb: str) -> List[str]:
    """Get allowed words for evaluation (consistent with training)"""
    allow = [w for w in candidates if w not in guessed]
    if attempts == 1 and last_fb in SECOND_GUESS_LOOKUP:
        g2 = SECOND_GUESS_LOOKUP[last_fb]
        if g2 not in guessed and g2 not in allow:
            allow.append(g2)
    if len(candidates) > 15 and attempts >= 2:
        for w in select_splitters(candidates, guessed, None, SPLIT_SAMPLE_POOL):
            if w not in guessed:
                allow.append(w)
    return list(dict.fromkeys(allow))

def play_wordle_rl(target_word: str, model: QNetwork, return_trace: bool = False):
    """Play Wordle using RL model"""
    candidates = TARGET_WORDS.copy()
    guessed = set(); attempts = 0
    cand_sizes = []; feedbacks = []; used_lookup = False; used_rl = False

    # First fixed guess
    guess = FIXED_FIRST
    fb = get_feedback(guess, target_word)
    candidates = filter_words(candidates, guess, fb)
    cand_sizes.append(len(candidates)); feedbacks.append(fb)
    if fb == 'GGGGG':
        return (1, True, cand_sizes, feedbacks, used_lookup, used_rl) if return_trace else (1, True)
    guessed.add(guess); attempts += 1; last_fb = fb

    while attempts < 6:
        allow = pick_allowed_words_eval(candidates, guessed, attempts, last_fb)
        # Force lookup in step 2
        if attempts == 1 and last_fb in SECOND_GUESS_LOOKUP and SECOND_GUESS_LOOKUP[last_fb] in allow:
            guess = SECOND_GUESS_LOOKUP[last_fb]; used_lookup = True
        else:
            x = torch.from_numpy(extract_state_features(candidates, attempts)).float().to(device).unsqueeze(0)
            with torch.no_grad():
                q = model(x)[0].cpu().numpy()
            if allow:
                idxs = [WORD2IDX[w] for w in allow]
                best = max(idxs, key=lambda i: q[i])
                guess = IDX2WORD[best]
            else:
                left = [w for w in GUESS_WORDS if w not in guessed]
                guess = random.choice(left) if left else random.choice(GUESS_WORDS)
            used_rl = True

        fb = get_feedback(guess, target_word)
        candidates = filter_words(candidates, guess, fb)
        cand_sizes.append(len(candidates)); feedbacks.append(fb)
        if fb == 'GGGGG':
            return (attempts + 1, True, cand_sizes, feedbacks, used_lookup, used_rl) if return_trace else (attempts + 1, True)
        guessed.add(guess); attempts += 1; last_fb = fb

    return (6, False, cand_sizes, feedbacks, used_lookup, used_rl) if return_trace else (6, False)

def evaluate_model(model: QNetwork, games: int = 3000, seed: int = 1112, description: str = "RL Solver"):
    """Evaluate RL model performance"""
    rng = random.Random(seed)

    attempts_list = []
    success_list  = []
    rl_used_list  = []
    runtimes      = []

    t0 = time.perf_counter()
    for i in range(games):
        tgt = rng.choice(TARGET_WORDS)

        t_game = time.perf_counter()
        ret = play_wordle_rl(tgt, model, return_trace=True)
        if ret is None:
            raise RuntimeError("play_wordle_rl returned None — expected a 6-tuple when return_trace=True")
        try:
            att, ok, _, _, _, used_rl = ret
        except Exception as e:
            raise RuntimeError(f"Unexpected return from play_wordle_rl: {type(ret)} -> {ret}") from e

        runtimes.append(time.perf_counter() - t_game)
        attempts_list.append(att)
        success_list.append(1 if ok else 0)
        rl_used_list.append(1 if used_rl else 0)

        if (i + 1) % 500 == 0:
            print(f"{description}: {i + 1}/{games}  time {time.perf_counter() - t0:.1f}s")

    # --- Summary statistics ---
    n = games
    avg_att = float(np.mean(attempts_list))
    att_lo, att_hi = bootstrap_mean_ci(attempts_list, n_boot=5000, alpha=0.05, seed=seed)

    succ_cnt  = int(np.sum(success_list))
    succ_rate = succ_cnt / n
    sr_lo, sr_hi = wilson_ci(succ_cnt, n, z=1.96)

    rl_cnt   = int(np.sum(rl_used_list))
    rl_rate  = rl_cnt / n
    rl_lo, rl_hi = wilson_ci(rl_cnt, n, z=1.96)

    rt_mean   = float(np.mean(runtimes))
    rt_median = float(np.median(runtimes))
    rt_p90    = float(np.percentile(runtimes, 90))

    # --- Print results (with 95% CI) ---
    print(f"\n==== {description} Results ====")
    print(f"Average attempts: {avg_att:.2f}  (95% CI: {att_lo:.2f}–{att_hi:.2f})")
    print(f"Success rate: {succ_rate*100:.2f}%  (95% CI: {sr_lo*100:.2f}–{sr_hi*100:.2f}%)")
    print(f"RL usage rate: {rl_rate*100:.2f}%  (95% CI: {rl_lo*100:.2f}–{rl_hi*100:.2f}%)")
    print(f"Single game inference time: mean={rt_mean:.4f}s, median={rt_median:.4f}s, P90={rt_p90:.4f}s")
    print("[Methodology] Average attempts count failures as 6; success/usage rates use Wilson 95% CI; average attempts use bootstrap 95% CI; time is inference time, not including training.")

    return {
        "games": n,
        "avg_attempts": avg_att,
        "avg_attempts_ci": (att_lo, att_hi),
        "success_rate": succ_rate,
        "success_rate_ci": (sr_lo, sr_hi),
        "rl_usage_rate": rl_rate,
        "rl_usage_rate_ci": (rl_lo, rl_hi),
        "runtime_mean": rt_mean,
        "runtime_median": rt_median,
        "runtime_p90": rt_p90,
    }    
def load_trained_model(path="wordle_dqn_enhanced.pth"):
    """Load a trained model"""
    # Input dimension based on your feature construction (157), can also hardcode 157
    input_dim = len(extract_state_features(TARGET_WORDS, 0))
    model = QNetwork(input_dim, VOCAB_SIZE, hidden=512).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for binomial proportion, returns (lo, hi) in [0,1]."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2*n)) / denom
    margin = (z / denom) * math.sqrt((phat*(1 - phat) / n) + (z**2) / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)

def bootstrap_mean_ci(data, n_boot: int = 3000, alpha: float = 0.05, seed: int = 1112):
    """Nonparametric bootstrap 95% CI for mean, returns (lo, hi)."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1 - alpha/2))
    return lo, hi
# =====================
# Main Process
# =====================
def main():
    # 1) Collect 30k demonstrations (with hard word oversampling & diversification)
    S, A, R, S2, D, allow_s, allow_s2 = collect_training_data(
        num_games=30000, seed=SEED, hard_ratio=0.2, diversify_prob=0.7
    )
    ds = WordleDataset(S, A, R, S2, D, allow_s, allow_s2)

    # 2) Train Double DQN (masking + BC + CQL)
    input_dim = S.shape[1]
    model = QNetwork(input_dim, VOCAB_SIZE, hidden=512)
    model = train_double_dqn_masked(model, ds, TrainCfg())
    torch.save(model.state_dict(), "wordle_dqn_enhanced.pth")
    print("Model training completed and saved -> wordle_dqn_enhanced.pth")

    # 3) Evaluate 3k games (epsilon=0, forced lookup in step 2)
    evaluate_model(model, games=3000, seed=1112, description="RL Solver")

if __name__ == "__main__":
    main()