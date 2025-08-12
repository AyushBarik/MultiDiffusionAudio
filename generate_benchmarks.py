#!/usr/bin/env python3
"""
MultiDiffusion Benchmark Audio Generation Script
Generates audio files with different hyperparameter combinations
Run this in the 'audioldm' environment
"""

import os
import sys
import torch
import numpy as np
import time
import json
from pathlib import Path

# Add AudioLDM to path
sys.path.append('./AudioLDM')
sys.path.append('./AudioLDM/audioldm')

try:
    from audioldm import build_model
    from audioldm.latent_diffusion.ddim import DDIMSampler
    from audioldm.pipeline import duration_to_latent_t_size
    from AudioLDM.utils.utils import *
    import soundfile as sf
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure you're in the 'audioldm' environment and running from the audioldm root directory")
    sys.exit(1)

def sliding_window_vae_decode(model, latents, window_size=256, overlap_size=64):
    """
    Decode latents using sliding windows with overlap-add reconstruction.
    
    Args:
        model: AudioLDM model with VAE decoder
        latents: Input latent tensor [B, C, T, F]
        window_size: Size of each window in latent frames (default: 256 ~= 10s)
        overlap_size: Overlap between windows in frames (default: 64 ~= 2.5s)
    
    Returns:
        Reconstructed mel spectrogram
    """
    device = latents.device
    B, C, T, F = latents.shape
    
    # If input is smaller than window, use regular decoding
    if T <= window_size:
        return model.decode_first_stage(latents)
    
    # Calculate windows
    step_size = window_size - overlap_size
    num_windows = (T - overlap_size + step_size - 1) // step_size
    
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
    
    # Calculate output dimensions
    mel_scale_factor = windows[0].shape[2] / window_positions[0][2]  # mel_frames / latent_frames
    total_mel_frames = int(T * mel_scale_factor)
    mel_channels = windows[0].shape[1]
    mel_freq_bins = windows[0].shape[3]
    
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
    
    # Normalize by weight sum to complete overlap-add
    output_mel = output_mel / torch.clamp(weight_sum, min=1e-8)
    
    return output_mel

def generate_audio_with_params(model, overlap_percent, chunk_size, prompt_segments, duration=30.0):
    """Generate audio using MultiDiffusion with specified parameters"""
    # Set parameters
    ddim_steps = 200
    ddim_eta = 0.1
    unconditional_guidance_scale = 3.0
    chunk_frames = chunk_size
    overlap_frames = int(chunk_frames * overlap_percent)
    
    # Ensure minimum overlap for naive chunking
    if overlap_percent == 0:
        overlap_frames = 0
    
    # Calculate expected number of chunks
    latent_size = duration_to_latent_t_size(duration)  # Should be 768 for 30s
    if overlap_percent == 0:
        # For 0% overlap, chunks don't overlap
        num_chunks = (latent_size + chunk_frames - 1) // chunk_frames  # Ceiling division
    else:
        # For overlapping chunks: advance_step = chunk_frames - overlap_frames
        advance_step = chunk_frames - overlap_frames
        if advance_step > 0:
            num_chunks = (latent_size - overlap_frames + advance_step - 1) // advance_step
        else:
            num_chunks = 1  # Edge case
    
    print(f"  📐 Parameters: chunk_frames={chunk_frames}, overlap_frames={overlap_frames} ({overlap_percent*100:.0f}% overlap)")
    print(f"  📊 Expected chunks: {num_chunks} (latent_size={latent_size}, advance_step={chunk_frames - overlap_frames})")
    
    # Create embeddings
    model.cond_stage_model.embed_mode = "text"
    segment_embeddings = []
    for start_time, end_time, segment_prompt in prompt_segments:
        segment_emb = model.get_learned_conditioning([segment_prompt])
        segment_embeddings.append(segment_emb)
    
    unconditional_embedding = model.get_learned_conditioning([""])
    
    # Setup sampling
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    z = torch.randn(shape).to(model.device)
    
    # Generate samples
    start_time = time.time()
    with torch.no_grad():
        if overlap_percent == 0:
            # Use naive/clean chunking for 0% overlap
            samples = multidiffusion_sample_clean(
                sampler=sampler,
                shape=shape,
                conditioning=segment_embeddings[0],  # Use first segment for naive
                unconditional_conditioning=unconditional_embedding,
                unconditional_guidance_scale=unconditional_guidance_scale,
                eta=ddim_eta,
                x_T=z,
                S=ddim_steps,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames  # Use calculated overlap_frames
            )
        else:
            # Use temporal MultiDiffusion
            samples = multidiffusion_sample_temporal(
                sampler=sampler,
                shape=shape,
                segment_embeddings=segment_embeddings,
                prompt_segments=prompt_segments,
                unconditional_conditioning=unconditional_embedding,
                unconditional_guidance_scale=unconditional_guidance_scale,
                eta=ddim_eta,
                x_T=z,
                S=ddim_steps,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                duration=duration
            )
    
    diffusion_time = time.time() - start_time
    
    # Decode with sliding window VAE
    vae_start = time.time()
    mel_spectrogram = sliding_window_vae_decode(model, samples, window_size=256, overlap_size=64)
    vae_time = time.time() - vae_start
    
    # Generate waveform
    vocoder_start = time.time()
    waveform = model.mel_spectrogram_to_waveform(mel_spectrogram)
    if isinstance(waveform, torch.Tensor):
        if waveform.dim() > 1:
            waveform = waveform[0]
        waveform = waveform.squeeze()
    else:
        if waveform.ndim > 1:
            waveform = waveform[0]
        waveform = waveform.squeeze()
    
    vocoder_time = time.time() - vocoder_start
    
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Return results
    return {
        'waveform': waveform,
        'num_chunks': num_chunks,
        'diffusion_time': diffusion_time,
        'vae_time': vae_time,
        'vocoder_time': vocoder_time,
        'total_time': diffusion_time + vae_time + vocoder_time
    }

def main():
    print("🚀 AudioLDM MultiDiffusion Benchmark Generator")
    print("=" * 60)
    
    # Define hyperparameters to test
    hyperparameters = [
         (0.25, 256),    # Small chunks with 70% overlap
         (0.50, 256),    # Medium-small chunks with 70% overlap
         (0.65, 256),    # Standard chunks with 70% overlap
         (0.75, 256),    # Large chunks with 70% overlap
         (0.85, 256),    # Very large chunks with 70% overlap
         (0, 256),      # Control: naive chunking (no overlap)
    ]
    
    # Test prompt segments
    prompt_segments = [
        (0.0, 30.0, "90s rock song with electric guitar and heavy drums")
    ]
    
    # Full prompt for metadata
    full_prompt = "90s rock song with electric guitar and heavy drums"
    duration = 30.0
    
    # Output setup
    output_folder = "benchaudios"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"📂 Output folder: {output_folder}")
    print(f"🎵 Test prompt: {full_prompt}")
    print(f"⏱️  Duration: {duration}s")
    print(f"🧪 Testing {len(hyperparameters)} hyperparameter combinations")
    print()
    
    # Load model
    print("📦 Loading AudioLDM model...")
    model = build_model(model_name="audioldm-m-full")
    sr = 16000
    print(f"✅ Model loaded successfully")
    print()
    
    # Metadata storage
    metadata = {
        'full_prompt': full_prompt,
        'prompt_segments': prompt_segments,
        'duration': duration,
        'sample_rate': sr,
        'generated_files': []
    }
    
    # Generate audio for each hyperparameter combination
    for i, (overlap_percent, chunk_size) in enumerate(hyperparameters):
        print(f"🔄 Generating {i+1}/{len(hyperparameters)}: overlap={overlap_percent*100:.0f}%, chunk_size={chunk_size}")
        
        try:
            # Generate audio (quiet mode)
            result = generate_audio_with_params(
                model, overlap_percent, chunk_size, prompt_segments, duration
            )
            
            # Save audio
            filename = f"multi_{overlap_percent:.2f}_{chunk_size}.wav"
            audio_path = os.path.join(output_folder, filename)
            sf.write(audio_path, result['waveform'], sr)
            
            # Add to metadata
            file_metadata = {
                'filename': filename,
                'overlap_percent': overlap_percent,
                'chunk_size': chunk_size,
                'expected_chunks': result['num_chunks'],
                'diffusion_time': result['diffusion_time'],
                'vae_time': result['vae_time'],
                'vocoder_time': result['vocoder_time'],
                'total_time': result['total_time'],
                'audio_length': len(result['waveform']) / sr
            }
            metadata['generated_files'].append(file_metadata)
            
            print(f"  ✅ Done in {result['total_time']:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            # Add failed entry to metadata
            file_metadata = {
                'filename': f"FAILED_{overlap_percent:.2f}_{chunk_size}.wav",
                'overlap_percent': overlap_percent,
                'chunk_size': chunk_size,
                'error': str(e),
                'diffusion_time': 0,
                'vae_time': 0,
                'vocoder_time': 0,
                'total_time': 0,
                'audio_length': 0
            }
            metadata['generated_files'].append(file_metadata)
    
    # Save metadata
    metadata_path = os.path.join(output_folder, 'generation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("🏁 GENERATION COMPLETE")
    print(f"📂 Audio files saved to: {output_folder}/")
    print(f"📋 Metadata saved to: {metadata_path}")
    print()
    
    # Print latency summary table
    print("⏱️  LATENCY SUMMARY")
    print("=" * 90)
    print(f"{'File':<25} {'Overlap%':<10} {'Chunk':<8} {'Chunks':<8} {'Diffusion':<11} {'VAE':<8} {'Vocoder':<9} {'Total':<8}")
    print("-" * 90)
    
    for file_info in metadata['generated_files']:
        if not file_info['filename'].startswith('FAILED_'):
            print(f"{file_info['filename']:<25} {file_info['overlap_percent']*100:<10.0f} {file_info['chunk_size']:<8} {file_info['expected_chunks']:<8} {file_info['diffusion_time']:<11.1f} {file_info['vae_time']:<8.1f} {file_info['vocoder_time']:<9.1f} {file_info['total_time']:<8.1f}")
    
    print("=" * 90)

if __name__ == "__main__":
    main()