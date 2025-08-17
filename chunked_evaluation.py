#!/usr/bin/env python3
"""
Chunk long-form audio into 10s segments for proper evaluation
"""

import os
import torch
import torchaudio
import glob
import re
import tempfile
import shutil
from audioldm_eval import EvaluationHelper

def chunk_audio_for_evaluation(audio_dir, output_dir, chunk_duration=10.0, sample_rate=16000):
    """Chunk audio files using RANDOM sampling to preserve statistical independence"""
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    chunk_count = 0
    file_mappings = {}
    
    print(f"📂 Chunking {len(audio_files)} audio files from {audio_dir}")
    
    for audio_file in audio_files:
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Extract base name for pairing (remove any existing chunk suffix)
        # e.g., "wav10_chunk_001" -> "wav10"
        base_match = re.match(r'^(.*?)(?:_chunk_\d+)?$', basename)
        if base_match:
            base_name = base_match.group(1)
        else:
            base_name = basename
            
        if base_name not in file_mappings:
            file_mappings[base_name] = []
        
        # Load audio
        waveform, sr = torchaudio.load(audio_file)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Calculate chunk size in samples
        chunk_samples = int(chunk_duration * sample_rate)
        total_samples = waveform.shape[1]
        
        print(f"  📄 {basename}: {total_samples/sample_rate:.1f}s → ", end="")
        
        # Create chunks using RANDOM sampling instead of sequential
        file_chunk_count = 0
        
        # Only proceed if audio is long enough for at least one full chunk
        if total_samples < chunk_samples:
            print(f"skipped (too short)")
            continue
            
        # Calculate how many chunks we can extract
        max_start_positions = max(1, (total_samples - chunk_samples) // (chunk_samples // 2))
        
        # Generate random start positions to ensure statistical independence
        import random
        random.seed(42)  # For reproducibility
        
        # Sample random positions, ensuring chunks don't overlap too much
        start_positions = []
        attempts = 0
        max_attempts = max_start_positions * 3  # Allow some retries
        
        while len(start_positions) < max_start_positions and attempts < max_attempts:
            start = random.randint(0, total_samples - chunk_samples)
            # Ensure minimum distance between chunks (at least 25% of chunk size)
            min_distance = chunk_samples // 4
            if not any(abs(start - pos) < min_distance for pos in start_positions):
                start_positions.append(start)
            attempts += 1
        
        # Sort for consistent ordering but maintain statistical independence
        start_positions.sort()
        
        for start_sample in start_positions:
            end_sample = start_sample + chunk_samples
            chunk = waveform[:, start_sample:end_sample]
            
            # Save chunk
            chunk_filename = f"{base_name}_chunk_{file_chunk_count:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            torchaudio.save(chunk_path, chunk, sample_rate)
            
            file_mappings[base_name].append(chunk_filename)
            
            file_chunk_count += 1
            chunk_count += 1
        
        print(f"{file_chunk_count} chunks")
    
    print(f"✅ Created {chunk_count} total chunks in {output_dir}")
    return chunk_count, file_mappings

def align_chunks_for_kl(ref_mappings, gen_mappings, temp_ref_dir, temp_gen_dir):
    """Ensure matched pairs for KL divergence calculation"""
    print("\n🔄 Aligning chunks for KL divergence...")
    
    # Find common base names
    common_bases = set(ref_mappings.keys()) & set(gen_mappings.keys())
    print(f"Found {len(common_bases)} common audio files")
    
    aligned_ref_dir = "temp_aligned_ref"  
    aligned_gen_dir = "temp_aligned_gen"
    os.makedirs(aligned_ref_dir, exist_ok=True)
    os.makedirs(aligned_gen_dir, exist_ok=True)
    
    pair_count = 0
    
    for base_name in common_bases:
        ref_chunks = ref_mappings[base_name]
        gen_chunks = gen_mappings[base_name]
        
        # Take minimum number of chunks to ensure pairs
        min_chunks = min(len(ref_chunks), len(gen_chunks))
        
        for i in range(min_chunks):
            # Copy with standardized naming
            ref_src = os.path.join(temp_ref_dir, ref_chunks[i])
            gen_src = os.path.join(temp_gen_dir, gen_chunks[i])
            
            pair_name = f"{base_name}_pair_{i:03d}.wav"
            ref_dst = os.path.join(aligned_ref_dir, pair_name)
            gen_dst = os.path.join(aligned_gen_dir, pair_name)
            
            shutil.copy2(ref_src, ref_dst)
            shutil.copy2(gen_src, gen_dst)
            pair_count += 1
    
    print(f"✅ Created {pair_count} aligned pairs")
    return aligned_ref_dir, aligned_gen_dir, pair_count

def evaluate_chunked_audio():
    """Main evaluation function with chunking"""
    print("🎵 AudioLDM Evaluation with Audio Chunking")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Define paths
    ref_dir = "chunked_ref"
    gen_dir = "artifacts/val/novel/config_7"
    
    # Create temporary directories for chunks
    temp_ref_dir = "temp_chunked_ref"
    temp_gen_dir = "temp_chunked_gen"
    
    try:
        print("\n🔪 CHUNKING AUDIO FILES...")
        
        # Chunk reference audio
        ref_chunks, ref_mappings = chunk_audio_for_evaluation(ref_dir, temp_ref_dir)
        
        # Chunk generated audio  
        gen_chunks, gen_mappings = chunk_audio_for_evaluation(gen_dir, temp_gen_dir)
        
        print(f"\n📊 Created {ref_chunks} reference chunks and {gen_chunks} generated chunks")
        
        # Align chunks for KL calculation
        aligned_ref_dir, aligned_gen_dir, pair_count = align_chunks_for_kl(
            ref_mappings, gen_mappings, temp_ref_dir, temp_gen_dir
        )
        
        # Initialize evaluator
        print("\n🔧 Initializing EvaluationHelper with CNN14/PANN backbone...")
        evaluator = EvaluationHelper(16000, device, backbone="cnn14")
        print("✅ EvaluationHelper initialized successfully!")
        
        print("\n🧮 Calculating metrics on aligned chunked audio...")
        metrics = evaluator.calculate_metrics(aligned_gen_dir, aligned_ref_dir, same_name=True)
        
        print("\n� CHUNKED EVALUATION RESULTS:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary directories
        print(f"\n🧹 Cleaning up temporary directories...")
        if os.path.exists("temp_aligned_ref"):
            shutil.rmtree("temp_aligned_ref")
            print(f"  ✅ Removed temp_aligned_ref")
        if os.path.exists("temp_aligned_gen"):
            shutil.rmtree("temp_aligned_gen")
            print(f"  ✅ Removed temp_aligned_gen")
        if os.path.exists(temp_ref_dir):
            shutil.rmtree(temp_ref_dir)
            print(f"  ✅ Removed {temp_ref_dir}")
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)
            print(f"  ✅ Removed {temp_gen_dir}")

if __name__ == "__main__":
    evaluate_chunked_audio()
