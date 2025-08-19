import numpy as np
from pathlib import Path
import librosa
import torch
from panns_inference import AudioTagging

# ================= CONFIG =================
REF_DIR = "chunked_ref"
GEN_DIR = "artifacts/val/novel/config_8"

SR = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8
KL_BATCH = 8

# Choose which metric(s) to print
DO_KL_SOFTMAX = True       # "KL" in audioldm_eval
DO_KL_SIGMOID = True       # "KL_Sigmoid" in audioldm_eval
# ==========================================

def _list_pairs(d1, d2):
    ref = {p.name: p for p in Path(d1).rglob("*.wav")}
    gen = {p.name: p for p in Path(d2).rglob("*.wav")}
    names = sorted(set(ref) & set(gen))
    if not names:
        raise RuntimeError("No paired WAVs with matching filenames.")
    return [(ref[n], gen[n]) for n in names]

def _load_batch_trim(pairs, sr=SR):
    xs_r, xs_g = [], []
    for r, g in pairs:
        xr, _ = librosa.load(r, sr=sr, mono=True)
        xg, _ = librosa.load(g, sr=sr, mono=True)
        L = min(len(xr), len(xg))
        xs_r.append(xr[:L]); xs_g.append(xg[:L])
    Lb = max(len(x) for x in xs_r)
    pad = lambda x: np.pad(x, (0, Lb - len(x))) if len(x) < Lb else x
    Xr = np.stack([pad(x) for x in xs_r], axis=0)
    Xg = np.stack([pad(x) for x in xs_g], axis=0)
    return Xr, Xg

def _safe_softmax(logits, axis=-1):
    # numerically stable softmax
    z = logits - np.max(logits, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.maximum(ez.sum(axis=axis, keepdims=True), EPS)

def _safe_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _normalize(p):
    s = np.maximum(p.sum(axis=-1, keepdims=True), EPS)
    return p / s

def _kl_pq(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def run_kl():
    pairs = _list_pairs(REF_DIR, GEN_DIR)
    at = AudioTagging(checkpoint_path=None, device=DEVICE)

    kl_vals, kl_sig_vals = [], []

    for i in range(0, len(pairs), KL_BATCH):
        batch = pairs[i:i+KL_BATCH]
        Xr, Xg = _load_batch_trim(batch, sr=SR)

        # PANNs returns sigmoid probs; convert to numpy
        pr_sig, _ = at.inference(Xr)  # [B, C], sigmoid probs
        pg_sig, _ = at.inference(Xg)
        pr_sig = np.asarray(pr_sig, dtype=np.float64)
        pg_sig = np.asarray(pg_sig, dtype=np.float64)

        # Recover "logits" via logit transform
        pr_logits = np.log(np.clip(pr_sig, EPS, 1.0 - EPS)) - np.log(np.clip(1.0 - pr_sig, EPS, 1.0 - EPS))
        pg_logits = np.log(np.clip(pg_sig, EPS, 1.0 - EPS)) - np.log(np.clip(1.0 - pg_sig, EPS, 1.0 - EPS))

        if DO_KL_SOFTPACK := DO_KL_SOFTMAX:
            # KL (softmax over logits)
            pr_sm = _safe_softmax(pr_logits, axis=-1)
            pg_sm = _safe_softmax(pg_logits, axis=-1)
            kl_vals.extend(_kl_pq(pr_sm, pg_sm).tolist())

        if DO_KL_SIGMOID:
            # KL_Sigmoid (sigmoid over logits) -> normalize to simplex for KL
            pr_sg = _normalize(_safe_sigmoid(pr_logits))
            pg_sg = _normalize(_safe_sigmoid(pg_logits))
            kl_sig_vals.extend(_kl_pq(pr_sg, pg_sg).tolist())

    if DO_KL_SOFTMAX and kl_vals:
        v = np.asarray(kl_vals, dtype=np.float64)
        print(f"KL (softmax over logits) mean: {v.mean():.6f}, std: {v.std():.6f}, n={len(v)}")

    if DO_KL_SIGMOID and kl_sig_vals:
        v = np.asarray(kl_sig_vals, dtype=np.float64)
        print(f"KL_Sigmoid (sigmoid over logits) mean: {v.mean():.6f}, std: {v.std():.6f}, n={len(v)}")

if __name__ == "__main__":
    run_kl()