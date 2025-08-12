#!/usr/bin/env python3
"""
Prepare train/val/test splits for audio evaluation

Creates a reproducible split file from the dataset.
Run this in benchenv (no AudioLDM dependencies needed).
"""

import os
import json
import argparse
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset splits')
    parser.add_argument('--data', required=True, help='Path to data.json')
    parser.add_argument('--audio-dir', required=True, help='Directory containing wav files')
    parser.add_argument('--val-size', type=int, default=10, help='Validation set size')
    parser.add_argument('--test-size', type=int, default=40, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', required=True, help='Output splits.json path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🔧 PREPARING DATASET SPLITS")
    print(f"📂 Loading dataset from {args.data}")
    
    # Load dataset
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Filter samples with existing audio files
    valid_samples = []
    missing_files = []
    
    for sample in data['data']:
        audio_path = os.path.join(args.audio_dir, f"wav{sample['id']}.wav")
        if os.path.exists(audio_path):
            valid_samples.append({
                'id': sample['id'],
                'caption': sample['caption'],
                'duration': sample['duration'],
                'audio_path': audio_path
            })
        else:
            missing_files.append(f"wav{sample['id']}.wav")
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    
    print(f"Found {len(valid_samples)} samples with existing audio files")
    
    # Set seed and shuffle
    random.seed(args.seed)
    random.shuffle(valid_samples)
    
    # Split data
    val_samples = valid_samples[:args.val_size]
    test_samples = valid_samples[args.val_size:args.val_size + args.test_size]
    
    # Create splits info
    splits = {
        'val_ids': [s['id'] for s in val_samples],
        'test_ids': [s['id'] for s in test_samples],
        'val_samples': val_samples,
        'test_samples': test_samples,
        'seed': args.seed,
        'val_size': len(val_samples),
        'test_size': len(test_samples),
        'total_available': len(valid_samples),
        'missing_files': missing_files
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Save splits
    with open(args.out, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✅ Split data: {len(val_samples)} val, {len(test_samples)} test samples")
    print(f"💾 Saved splits to {args.out}")

if __name__ == '__main__':
    main()
