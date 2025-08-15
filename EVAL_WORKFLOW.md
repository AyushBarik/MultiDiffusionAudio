# Long-form Audio Generation & Evaluation (Two-Environment Workflow)

This guide explains how to run long-form audio generation and evaluation using two separate Python environments:
- ldmenv: generation only (AudioLDM + model deps) 
- benchenv: scoring only (CLAP, PANNs, Gemini, SciPy, etc.)

The existing scripts work with a simpler layout - no "artifacts" directory needed.

Contents
- Environments
- Current file layout
- Quick start
- Step-by-step
- Troubleshooting

Environments
- ldmenv (AudioLDM runtime):
  - Use to generate audio only.
  - Example activation (zsh):
    source ldmenv/bin/activate
- benchenv (metrics runtime):
  - Use to compute CLAP/FD/IS/KL and Gemini.
  - Example activation (zsh):
    source benchenv/bin/activate

Current file layout (what actually exists)
- splits.json (validation/test splits)
- val_gens/config_{id}/ (validation audio from tuning.py)
- tuning_results.json (best config from tuning.py)
- tuning_config_summary.csv
- tuning_detailed_results.csv
- eval.py (final evaluation script)
- Output from eval.py:
  - eval_per_sample.csv
  - eval_pairwise.csv
  - eval_aggregates.csv
  - eval_gemini.csv (if Gemini API used)
  - eval_results.json

Quick start (TL;DR)
1) Splits already exist in splits.json
2) In ldmenv: Run tuning.py to generate validation WAVs and find best config
3) In ldmenv: Generate test audio using notebooks or modify tuning.py
4) In benchenv: Run eval.py to compare novel vs baseline

Prerequisites
- Dataset audio present at: AudioSet/downloaded_audio/wav{id}.wav
- Exactly 50 WAVs expected. If something is missing (e.g., wav35.wav), fix or re-download/convert before splitting.
- Sanity check:
  ls AudioSet/downloaded_audio | wc -l  # should be 50

Step-by-step

0) Prepare splits (benchenv)
- Purpose: create a reproducible split file with val/test IDs.
- Command:
  python prepare_splits.py \
    --data AudioSet/data.json \
    --audio-dir AudioSet/downloaded_audio \
    --val-size 10 \
    --test-size 40 \
    --seed 42 \
    --out artifacts/splits.json
- Notes:
  - If a file is missing on disk, it will be excluded and sizes adjust accordingly.
  - With all 50 present, you will get 10 val + 40 test.

1) Generate validation audio for tuning (ldmenv)
- Purpose: generate validation WAVs for each grid config (novel/MultiDiffusion).
- Output: artifacts/val/novel/config_{id}/wav{id}.wav
- Example:
  python gen_tuning.py \
    --splits splits.json \
    --use-sample-durations \
    --overlap-percents 0.25 0.5 0.75 \
    --chunk-frames 64 128 256 \
    --ddim-steps 200 \
    --out-root artifacts/val/novel > gen_tuning.log 2>&1
- What it does:
  - For every (overlap_percent, chunk_frames) pair, generates WAVs for all val_ids.
  - Uses individual sample durations from data.json for fair comparison.
  - Saves timing data and config info.
  - Deterministic generation (seeds set).

2) Score tuning results and pick best config (benchenv)
- Purpose: compute CLAP/FD/IS/KL and required Gemini evaluation on validation WAVs and choose the best config.
- Outputs:
  - artifacts/tuning_results.json (includes best_config)
  - artifacts/tuning_config_summary.csv
  - artifacts/tuning_detailed_results.csv
  - artifacts/tuning_gemini_results.csv
- Example:
  python score_tuning.py \
    --val-root artifacts/val/novel \
    --splits splits.json \
    --out-root artifacts \
    --gemini-api-key "$GEMINI_API_KEY"
- Notes:
  - No AudioLDM imports; evaluation-only dependencies.
  - Reference audio resampled to 16kHz to match AudioLDM output.
  - Gemini API key is required, not optional.
  - CSVs contain per-config aggregates and per-sample details.

3) Generate test audio for best config and baseline (ldmenv)
- Purpose:
  - Generate “novel” WAVs using the best tuning config for all test_ids.
  - Generate “baseline” WAVs (naive chunking) for the same test_ids.
- Outputs:
  - artifacts/test/novel/wav{id}.wav
  - artifacts/test/baseline/wav{id}.wav
- Example:
  python gen_test.py \
    --splits artifacts/splits.json \
    --best artifacts/tuning_results.json \
    --duration 100 \
    --out-root artifacts/test
- Notes:
  - Best config is read from artifacts/tuning_results.json (written in Step 2).
  - Baseline uses naive chunking; novel uses MultiDiffusion with the best config.

4) Score final test results (benchenv)
- Purpose: compute CLAP/FD/IS/KL and Gemini (optional) for test novel vs. baseline.
- Outputs (examples):
  - artifacts/eval_per_sample.csv
  - artifacts/eval_pairwise.csv
  - artifacts/eval_aggregates.csv
  - artifacts/eval_gemini.csv (if API key provided)
  - artifacts/eval_results.json
- Example:
  export GEMINI_API_KEY="YOUR_KEY"  # optional for Gemini
  python score_test.py \
    --test-root artifacts/test \
    --splits artifacts/splits.json \
    --out-root artifacts \
    --gemini-api-key "$GEMINI_API_KEY"
- Notes:
  - If no API key is supplied, Gemini evaluation is skipped.
  - CSVs contain exhaustive per-sample and aggregate metrics.

Reproducibility
- Seeds are set in generation and scoring where applicable.
- artifacts/splits.json records IDs and seed.
- artifacts/tuning_results.json records the chosen config and grid.
- Each generation script should emit a small manifest per run (optional).

Troubleshooting
- Different Python versions:
  - Keep ldmenv and benchenv separate. Do not import scoring libs in generation or vice versa.
- Missing audio files:
  - Ensure all 50 files exist in AudioSet/downloaded_audio as wav{id}.wav.
  - Re-run prepare_splits.py to refresh splits after fixing files.
- GPU memory (OOM) during generation:
  - Reduce --chunk-frames, increase overlap, reduce --duration, or lower DDIM steps.
  - Close other GPU workloads.
- Slow downloads for PANNs/CLAP weights:
  - First run fetches checkpoints; subsequent runs are cached.
- File count sanity checks:
  - Validation WAVs for each config:
    find artifacts/val/novel/config_* -name 'wav*.wav' | wc -l  # Expect val_size * num_configs
  - Test WAVs:
    ls artifacts/test/novel | wc -l      # Expect test_size
    ls artifacts/test/baseline | wc -l   # Expect test_size

FAQ
- Why did I previously see 39 test items?
  - One file (e.g., wav35.wav) was missing; prepare_splits excluded it. After adding it, you should see 10 val + 40 test.
- Can I change the grid?
  - Yes; pass different --overlap-percents and --overlap-sizes to gen_tuning.py. Then re-run score_tuning.py to select a new best config.
- Can I use notebooks for generation instead of scripts?
  - Yes. Use:
    - AudioLDM/Multidiffusion_audioLDM_inference.ipynb (novel)
    - AudioLDM/NaiveChunking_audioLDM_inference.ipynb (baseline)
    Save outputs to the artifacts paths above, then run scoring steps in benchenv.

Notes
- The following scripts are expected by this workflow:
  - prepare_splits.py (benchenv)
  - gen_tuning.py, gen_test.py (ldmenv)
  - score_tuning.py, score_test.py (benchenv)
- If any are missing, use the notebooks for generation and existing scoring scripts as a temporary path, or request patches to add them.
