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
import sys

# Add the audioldm_eval path to monkey patch the padding function
sys.path.append('./benchenv/lib/python3.10/site-packages/audioldm_eval')

# Import first, then monkey patch
from audioldm_eval import EvaluationHelper
import audioldm_eval.datasets.load_mel

# Now monkey patch the padding function to do nothing - this is causing the metrics to be unreliable
from audioldm_eval.datasets.load_mel import pad_short_audio
def no_pad(audio, min_samples=32000):
    """Return audio as-is, no padding - this fixes the metric reliability issues"""
    return audio

# Replace the function globally
audioldm_eval.datasets.load_mel.pad_short_audio = no_pad


def chunk_audio_for_evaluation(audio_dir, output_dir, chunk_duration=10.0, sample_rate=16000):
    """
    Sequential fixed-window chunking with zero-padding to match AudioLDM evaluation.
    Every audio is split into non-overlapping 10s windows.
    If the last chunk is shorter, it's padded with zeros.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    chunk_count = 0
    file_mappings = {}

    print(f"📂 Fixed-window chunking {len(audio_files)} files from {audio_dir}")

    for audio_file in audio_files:
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        base_name = re.match(r'^(.*?)(?:_chunk_\d+)?$', basename).group(1)

        if base_name not in file_mappings:
            file_mappings[base_name] = []

        waveform, sr = torchaudio.load(audio_file)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

        chunk_samples = int(chunk_duration * sample_rate)
        total_samples = waveform.shape[1]

        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples  # ceil division

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]

            # Pad last chunk if needed
            if chunk.shape[1] < chunk_samples:
                pad_len = chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            chunk_filename = f"{base_name}_chunk_{i:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            torchaudio.save(chunk_path, chunk, sample_rate)

            file_mappings[base_name].append(chunk_filename)
            chunk_count += 1

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
        
        # DEBUG: Check what files we're actually evaluating
        print(f"\n🔍 DEBUG: Checking audio files before evaluation...")
        ref_files = glob.glob(os.path.join(aligned_ref_dir, "*.wav"))
        gen_files = glob.glob(os.path.join(aligned_gen_dir, "*.wav"))
        
        print(f"Reference files: {len(ref_files)}")
        print(f"Generated files: {len(gen_files)}")
        
        # Check first few files for duration
        for i, file_path in enumerate(ref_files[:3]):
            waveform, sr = torchaudio.load(file_path)
            duration = waveform.shape[1] / sr
            print(f"  Ref {i}: {os.path.basename(file_path)} → {duration:.1f}s ({waveform.shape[1]} samples)")
            
        for i, file_path in enumerate(gen_files[:3]):
            waveform, sr = torchaudio.load(file_path)
            duration = waveform.shape[1] / sr
            print(f"  Gen {i}: {os.path.basename(file_path)} → {duration:.1f}s ({waveform.shape[1]} samples)")
        
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
