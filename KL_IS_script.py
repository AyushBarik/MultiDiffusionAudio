#!/usr/bin/env python3
import torch
from audioldm_eval import EvaluationHelper

REF_DIR = "chunked_ref"
GEN_DIR = "artifacts/val/novel/config_9"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = EvaluationHelper(32000, device, backbone="cnn14")  

    metrics = evaluator.main(
        GEN_DIR, REF_DIR,
        limit_num=None
    )

    # Read from correct long-form keys that are actually returned
    is_mean = metrics.get("inception_score_mean")
    is_std  = metrics.get("inception_score_std") 
    kl_softmax = metrics.get("kullback_leibler_divergence_softmax")  # KL softmax - only works in paired mode

    print("=== RESULTS ===")
    if is_mean is not None:
        print(f"IS: {is_mean:.6f}" + (f" ± {is_std:.6f}" if is_std is not None else ""))
    if kl_softmax is not None:
        print(f"KL (softmax): {kl_softmax:.6f}")

if __name__ == "__main__":
    main()