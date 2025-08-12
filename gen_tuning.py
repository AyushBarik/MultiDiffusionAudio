#!/usr/bin/env python3
"""
Generate validation audio for hyperparameter tuning

Run this in ldmenv (AudioLDM environment).
Generates WAVs for all validation samples across grid configs.
"""

import os
import sys
import json
import argparse
import random
import itertools
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

# Add AudioLDM path
sys.path.append('AudioLDM')
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size

# Import MultiDiffusion helpers
from AudioLDM.utils.utils import multidiffusion_sample_temporal

def sliding_window_vae_decode(model, latents, window_size=256, overlap_size=64):
    """
    Decode latents using sliding windows with overlap-add reconstruction.
    Following the exact implementation from Multidiffusion_audioLDM_inference.ipynb
    """
    device = latents.device
    B, C, T, F = latents.shape
    
    print(f"  🪟 SLIDING WINDOW VAE: Input shape: {latents.shape}")
    print(f"  Window size: {window_size} frames (~{window_size/25.6:.1f}s)")
    print(f"  Overlap size: {overlap_size} frames (~{overlap_size/25.6:.1f}s)")
    
    # If input is smaller than window, use regular decoding
    if T <= window_size:
        print(f"  Input smaller than window - using regular decode")
        return model.decode_first_stage(latents)
    
    # Calculate windows
    step_size = window_size - overlap_size
    num_windows = (T - overlap_size + step_size - 1) // step_size
    
    print(f"  Step size: {step_size} frames")
    print(f"  Number of windows: {num_windows}")
    
    # Process each window
    windows = []
    window_positions = []
    
    for i in range(num_windows):
        start = i * step_size
        end = min(start + window_size, T)
        actual_window_size = end - start
        
        # Extract window
        window_latent = latents[:, :, start:end, :]
        
        # Decode window
        with torch.no_grad():
            window_mel = model.decode_first_stage(window_latent)
        
        windows.append(window_mel)
        window_positions.append((start, end, actual_window_size))
        
        print(f"    Window {i+1}: latent[{start}:{end}] -> mel shape {window_mel.shape}")
    
    # Calculate output dimensions
    mel_scale_factor = windows[0].shape[2] / window_positions[0][2]  # mel_frames / latent_frames
    total_mel_frames = int(T * mel_scale_factor)
    mel_channels = windows[0].shape[1]
    mel_freq_bins = windows[0].shape[3]
    
    print(f"  Mel scale factor: {mel_scale_factor:.2f}x")
    print(f"  Output mel shape: [{B}, {mel_channels}, {total_mel_frames}, {mel_freq_bins}]")
    
    # Initialize output tensor and weight accumulator
    output_mel = torch.zeros((B, mel_channels, total_mel_frames, mel_freq_bins), 
                            device=device, dtype=windows[0].dtype)
    weight_sum = torch.zeros_like(output_mel)
    
    # Create overlap-add weights (linear fade in/out)
    overlap_mel_size = int(overlap_size * mel_scale_factor)
    
    for i, (window_mel, (start, end, actual_size)) in enumerate(zip(windows, window_positions)):
        # Calculate mel positions
        mel_start = int(start * mel_scale_factor)
        mel_end = int(end * mel_scale_factor)
        actual_mel_size = window_mel.shape[2]
        
        # Create weight mask for this window
        weight_mask = torch.ones((1, 1, actual_mel_size, 1), device=device)
        
        # Apply fade-in at the beginning (except for first window)
        if i > 0 and overlap_mel_size > 0:
            fade_in_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_in = torch.linspace(0, 1, fade_in_size, device=device)
            weight_mask[:, :, :fade_in_size, :] = fade_in.view(1, 1, -1, 1)
        
        # Apply fade-out at the end (except for last window)
        if i < len(windows) - 1 and overlap_mel_size > 0:
            fade_out_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_out = torch.linspace(1, 0, fade_out_size, device=device)
            weight_mask[:, :, -fade_out_size:, :] = fade_out.view(1, 1, -1, 1)
        
        # Add weighted window to output
        mel_end_actual = mel_start + actual_mel_size
        output_mel[:, :, mel_start:mel_end_actual, :] += window_mel * weight_mask
        weight_sum[:, :, mel_start:mel_end_actual, :] += weight_mask
        
        print(f"    Added window {i+1} to output[{mel_start}:{mel_end_actual}]")
    
    # Normalize by weight sum to complete overlap-add
    output_mel = output_mel / torch.clamp(weight_sum, min=1e-8)
    
    return output_mel

def parse_args():
    parser = argparse.ArgumentParser(description='Generate validation audio for tuning')
    parser.add_argument('--splits', required=True, help='Path to splits.json')
    parser.add_argument('--duration', type=float, default=None, help='Fixed audio duration in seconds (optional)')
    parser.add_argument('--use-sample-durations', action='store_true', help='Use individual sample durations from data.json')
    parser.add_argument('--overlap-percents', nargs='+', type=float, default=[0.25, 0.5, 0.75],
                        help='Overlap percentages to test (fraction of chunk_frames)')
    parser.add_argument('--chunk-frames', nargs='+', type=int, default=[64, 128, 256],
                        help='Chunk sizes in frames to test')
    parser.add_argument('--ddim-steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--out-root', required=True, help='Output root directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def setup_environment(seed=42):
    """Setup random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_audio_multidiffusion(model, prompt, duration, overlap_percent, chunk_frames, ddim_steps=200):
    """Generate audio using MultiDiffusion"""
    import time
    
    setup_environment(42 + hash(prompt) % 1000)  # Deterministic per prompt
    
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    
    # Calculate chunks
    latent_size = duration_to_latent_t_size(duration)
    overlap_frames = int(chunk_frames * overlap_percent)
    advance_step = chunk_frames - overlap_frames
    
    num_chunks = (latent_size - overlap_frames + advance_step - 1) // advance_step
    
    print(f"DEBUG: total_frames={latent_size}, chunk_size={chunk_frames}, overlap_frames={overlap_frames}, advance_step={advance_step}")
    
    # Debug chunk layout
    chunks = []
    for i in range(num_chunks):
        start = i * advance_step
        end = min(start + chunk_frames, latent_size)
        if end > start:
            chunks.append((start, end))
            print(f"DEBUG: chunk {i+1}: ({start}, {end}) - frames: {end-start}")
    
    print(f"DEBUG: Created {len(chunks)} chunks covering frames 0-{latent_size}")
    
    # Set up embeddings and sampling (following notebook pattern)
    model.cond_stage_model.embed_mode = "text"
    
    # Create embeddings
    text_list = [prompt]
    text_embedding = model.get_learned_conditioning(text_list)
    unconditional_text = [""]
    unconditional_embedding = model.get_learned_conditioning(unconditional_text)
    
    # Create segment embeddings for temporal sampling
    segment_embeddings = [text_embedding]
    prompt_segments = [(0.0, duration, prompt)]
    
    # Set up sampling
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    z = torch.randn(shape).to(model.device)
    
    # Generate using MultiDiffusion temporal sampling
    with torch.no_grad():
        samples = multidiffusion_sample_temporal(
            sampler=sampler,
            shape=shape,
            segment_embeddings=segment_embeddings,
            prompt_segments=prompt_segments,
            unconditional_conditioning=unconditional_embedding,
            unconditional_guidance_scale=2.5,
            eta=0.0,
            x_T=z,
            S=ddim_steps,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            duration=duration
        )
        
        # Decode latent to mel spectrogram using sliding window (following notebook)
        print(f"🪟 USING SLIDING WINDOW VAE DECODER:")
        mel_spectrogram = sliding_window_vae_decode(model, samples, window_size=256, overlap_size=64)
        
        # Convert mel to waveform
        waveform = model.mel_spectrogram_to_waveform(mel_spectrogram)
        
        # Handle tensor shape (following notebook pattern)
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() > 1:
                waveform = waveform[0]  # Take first batch item
            waveform = waveform.squeeze()
        else:  # NumPy array
            if waveform.ndim > 1:
                waveform = waveform[0]
            waveform = waveform.squeeze()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    generation_time = time.time() - start_time
    memory_peak = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    memory_current = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    
    timing_stats = {
        'generation_time': generation_time,
        'memory_before': memory_before,
        'memory_peak': memory_peak,
        'memory_current': memory_current,
        'num_chunks': len(chunks),
        'time_per_second': generation_time / duration,
        'time_per_chunk': generation_time / len(chunks) if chunks else 0
    }
    
    return waveform, timing_stats

def main():
    args = parse_args()
    
    print("🔧 AUDIOLDM VALIDATION GENERATION")
    print("=" * 50)
    
    # Setup environment
    setup_environment(args.seed)
    
    # Load splits
    print(f"📂 Loading splits from {args.splits}")
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    # Load full dataset to get captions
    print(f"📂 Loading full dataset from AudioSet/data.json")
    with open('AudioSet/data.json', 'r') as f:
        dataset = json.load(f)
    
    # Create lookup by ID
    data_by_id = {str(sample['id']): sample for sample in dataset['data']}
    
    # Build validation samples from splits
    val_samples = []
    for val_id in splits['val_ids']:
        if val_id in data_by_id:
            sample = data_by_id[val_id]
            val_samples.append({
                'id': sample['id'],
                'caption': sample['caption'],
                'duration': sample['duration'],
                'audio_path': f"AudioSet/downloaded_audio/wav{sample['id']}.wav"
            })
        else:
            print(f"⚠️  Warning: ID {val_id} not found in dataset")
    
    print(f"Found {len(val_samples)} validation samples")
    
    # Initialize AudioLDM
    print("🤖 Initializing AudioLDM model...")
    model = build_model(model_name="audioldm-m-full")
    model.cond_stage_model.embed_mode = "text"
    
    # Create hyperparameter grid
    configs = []
    config_id = 0
    for overlap_percent, chunk_frames in itertools.product(args.overlap_percents, args.chunk_frames):
        config_id += 1
        configs.append({
            'id': config_id,
            'overlap_percent': overlap_percent,
            'chunk_frames': chunk_frames
        })
    
    print(f"🎛️  Grid search: {len(configs)} configurations")
    print(f"   Overlap percents: {args.overlap_percents}")
    print(f"   Chunk frames: {args.chunk_frames}")
    if args.use_sample_durations:
        print(f"   Using individual sample durations")
    else:
        print(f"   Fixed duration: {args.duration}s")
    print(f"   Validation samples: {len(val_samples)}")
    
    # Prepare timing data storage
    all_timing_data = []
    
    # Generate audio for each config
    for config in configs:
        config_id = config['id']
        overlap_percent = config['overlap_percent']
        chunk_frames = config['chunk_frames']
        overlap_frames = int(chunk_frames * overlap_percent)
        
        print(f"\n⚙️  Config {config_id}: chunk_frames={chunk_frames}, overlap_percent={overlap_percent} ({overlap_frames} frames)")
        
        # Create output directory
        config_dir = os.path.join(args.out_root, f"config_{config_id}")
        os.makedirs(config_dir, exist_ok=True)
        
        # Save config info
        config_info = {
            'config_id': config_id,
            'overlap_percent': overlap_percent,
            'chunk_frames': chunk_frames,
            'overlap_frames': overlap_frames,
            'ddim_steps': args.ddim_steps
        }
        with open(os.path.join(config_dir, 'config.json'), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        # Generate for each validation sample
        for sample in val_samples:
            sample_id = sample['id']
            prompt = sample['caption']
            
            # Determine duration
            if args.use_sample_durations:
                duration = sample['duration']
            else:
                duration = args.duration
            
            print(f"   Generating wav{sample_id}.wav ({duration:.1f}s): '{prompt[:50]}...'")
            
            try:
                # Generate audio with timing
                waveform, timing_stats = generate_audio_multidiffusion(
                    model=model,
                    prompt=prompt,
                    duration=duration,
                    overlap_percent=overlap_percent,
                    chunk_frames=chunk_frames,
                    ddim_steps=args.ddim_steps
                )
                
                # Verify audio was generated successfully
                if waveform is None or len(waveform) == 0:
                    raise ValueError(f"Generated waveform is empty or None")
                
                # Save audio
                output_path = os.path.join(config_dir, f"wav{sample_id}.wav")
                sf.write(output_path, waveform, 16000)
                
                # Verify file was saved successfully
                if not os.path.exists(output_path):
                    raise ValueError(f"Failed to save audio file: {output_path}")
                
                print(f"      ✅ Successfully saved: {output_path}")
                
                # Save timing data
                timing_record = {
                    'config_id': config_id,
                    'sample_id': sample_id,
                    'duration': duration,
                    'overlap_percent': overlap_percent,
                    'chunk_frames': chunk_frames,
                    **timing_stats
                }
                all_timing_data.append(timing_record)
                
                print(f"      ⏱️  {timing_stats['generation_time']:.1f}s gen time, {timing_stats['num_chunks']} chunks, {timing_stats['time_per_second']:.2f}s/s")
                
            except Exception as e:
                print(f"      ❌ CRITICAL ERROR generating wav{sample_id}.wav: {e}")
                print(f"      🛑 HALTING execution - audio generation failed!")
                import traceback
                traceback.print_exc()
                raise SystemExit(f"Generation failed for sample {sample_id}. Cannot continue without all validation samples.")
    
    # Save timing data
    timing_path = os.path.join(args.out_root, 'timing_data.json')
    with open(timing_path, 'w') as f:
        json.dump(all_timing_data, f, indent=2)
    
    print(f"\n✅ Validation generation completed!")
    print(f"💾 Audio saved to {args.out_root}/config_{{1..{len(configs)}}}/")
    print(f"⏱️  Timing data saved to {timing_path}")

if __name__ == '__main__':
    main()
