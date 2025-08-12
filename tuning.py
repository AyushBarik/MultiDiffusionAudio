#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for AudioLDM MultiDiffusion Method

This script performs grid search on key hyperparameters for the multidiffusion method
(novel inference) on validation samples to find optimal parameters.

Usage:
    python tuning.py --overlap_percents 0.25 0.50 0.75 --overlap_sizes 64 128 192

Features:
- Grid search over overlap percentages and overlap sizes (in latent frames)
- Evaluation on 10 validation samples 
- Metrics: CLAP, FD, IS, KL
- Composite scoring for best config selection
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
import torchaudio
from tqdm import tqdm
import laion_clap

# Add AudioLDM path
sys.path.append('AudioLDM')
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size

# Import MultiDiffusion helpers explicitly to avoid importing a different 'utils'
from AudioLDM.utils.utils import multidiffusion_sample_clean

# Ensure PANNs inference is available
try:
    from panns_inference import AudioTagging
except ImportError:
    print("Installing panns-inference...")
    os.system("pip install panns-inference")
    from panns_inference import AudioTagging

print("🔧 AUDIOLDM HYPERPARAMETER TUNING")
print("=" * 50)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AudioLDM MultiDiffusion Hyperparameter Tuning')
    parser.add_argument('--overlap_percents', nargs='+', type=float, default=[0.25, 0.50, 0.75],
                        help='List of overlap percentages to test (e.g., 0.25 0.50 0.75)')
    parser.add_argument('--overlap_sizes', nargs='+', type=int, default=[64, 128, 192], 
                        help='List of overlap sizes in latent frames to test (e.g., 64 128 192)')
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of validation samples to use')
    parser.add_argument('--test_size', type=int, default=40,
                        help='Number of test samples to use')
    parser.add_argument('--duration', type=float, default=100.0,
                        help='Duration of generated audio in seconds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def setup_environment(seed=42):
    """Setup random seeds and environment"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_and_split_data(data_path, audio_dir, val_size=10, test_size=40, seed=42):
    """
    Load dataset and split into validation and test sets
    
    Returns:
        val_samples: List of validation samples
        test_samples: List of test samples  
        splits: Dictionary with split information
    """
    print(f"📂 Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Filter to only include samples with existing audio files
    valid_samples = []
    for sample in data['data']:
        audio_path = os.path.join(audio_dir, f"wav{sample['id']}.wav")
        if os.path.exists(audio_path):
            valid_samples.append({
                'id': sample['id'],
                'caption': sample['caption'],
                'duration': sample['duration'],
                'audio_path': audio_path
            })
    
    print(f"Found {len(valid_samples)} samples with existing audio files")
    
    # Shuffle and split
    random.shuffle(valid_samples)
    val_samples = valid_samples[:val_size]
    test_samples = valid_samples[val_size:val_size + test_size]
    
    # Create splits dictionary
    splits = {
        'val_ids': [s['id'] for s in val_samples],
        'test_ids': [s['id'] for s in test_samples],
        'seed': seed,
        'val_size': len(val_samples),
        'test_size': len(test_samples)
    }
    
    # Save splits for reproducibility
    with open('splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✅ Split data: {len(val_samples)} val, {len(test_samples)} test samples")
    print(f"💾 Saved splits to splits.json")
    
    return val_samples, test_samples, splits

def initialize_models():
    """Initialize AudioLDM and evaluation models"""
    print("🤖 Initializing models...")
    
    # Initialize AudioLDM
    print("  - Loading AudioLDM model...")
    audioldm_model = build_model(model_name="audioldm-m-full")
    audioldm_model.cond_stage_model.embed_mode = "text"
    
    # Initialize CLAP for text-audio similarity
    print("  - Loading CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    # Load CLAP checkpoint if available
    clap_ckpt = 'CLAP90.14.pt'
    if os.path.exists(clap_ckpt):
        clap_model.load_ckpt(ckpt=clap_ckpt)
        print(f"    ✅ Loaded CLAP checkpoint: {clap_ckpt}")
    else:
        print(f"    ⚠️  CLAP checkpoint not found: {clap_ckpt}")
    
    # Initialize PANNs for audio feature extraction
    print("  - Loading PANNs model...")
    try:
        # Newer panns-inference versions don't accept model_type
        panns_model = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    except TypeError:
        # Fallback for older API (if any)
        panns_model = AudioTagging(checkpoint_path=None)
    
    print("✅ All models initialized")
    return audioldm_model, clap_model, panns_model

def generate_audio_multidiffusion(model, prompt, duration, overlap_percent, overlap_size, 
                                  chunk_frames=256, ddim_steps=200):
    """
    Generate audio using MultiDiffusion with specified overlap parameters
    
    Args:
        model: AudioLDM model
        prompt: Text prompt
        duration: Duration in seconds
        overlap_percent: Overlap percentage (0.0 to 1.0)
        overlap_size: Overlap size in frames
        chunk_frames: Size of each chunk in frames
        ddim_steps: Number of DDIM steps
    
    Returns:
        waveform: Generated audio waveform
    """
    # Prepare conditioning
    text_emb = model.get_learned_conditioning([prompt])
    unconditional_embedding = model.get_learned_conditioning([""])
    
    # Setup parameters
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    z = torch.randn(shape).to(model.device)
    
    # Calculate actual overlap frames from percentage if provided
    if overlap_percent > 0:
        calculated_overlap = int(chunk_frames * overlap_percent)
        actual_overlap = max(calculated_overlap, overlap_size)  # Use larger of two
    else:
        actual_overlap = overlap_size
    
    # Generate latent using MultiDiffusion
    with torch.no_grad():
        samples = multidiffusion_sample_clean(
            sampler=sampler,
            shape=shape,
            conditioning=text_emb,
            unconditional_conditioning=unconditional_embedding,
            unconditional_guidance_scale=3.0,
            eta=0.1,
            x_T=z,
            S=ddim_steps,
            chunk_frames=chunk_frames,
            overlap_frames=actual_overlap
        )
        
        # Decode to mel spectrogram using sliding window VAE
        mel_spectrogram = sliding_window_vae_decode(model, samples, window_size=256, overlap_size=64)
        
        # Convert to waveform
        waveform = model.mel_spectrogram_to_waveform(mel_spectrogram)
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() > 1:
                waveform = waveform[0]
            waveform = waveform.squeeze()
        else:
            if waveform.ndim > 1:
                waveform = waveform[0]
            waveform = waveform.squeeze()
    
    return waveform

def sliding_window_vae_decode(model, latents, window_size=256, overlap_size=64):
    """Sliding window VAE decoder from the notebook"""
    device = latents.device
    B, C, T, F = latents.shape
    
    if T <= window_size:
        return model.decode_first_stage(latents)
    
    step_size = window_size - overlap_size
    num_windows = (T - overlap_size + step_size - 1) // step_size
    
    windows = []
    window_positions = []
    
    for i in range(num_windows):
        start = i * step_size
        end = min(start + window_size, T)
        actual_window_size = end - start
        
        window_latent = latents[:, :, start:end, :]
        
        with torch.no_grad():
            window_mel = model.decode_first_stage(window_latent)
        
        windows.append(window_mel)
        window_positions.append((start, end, actual_window_size))
    
    # Calculate output dimensions
    mel_scale_factor = windows[0].shape[2] / window_positions[0][2]
    total_mel_frames = int(T * mel_scale_factor)
    mel_channels = windows[0].shape[1]
    mel_freq_bins = windows[0].shape[3]
    
    # Initialize output tensor and weight accumulator
    output_mel = torch.zeros((B, mel_channels, total_mel_frames, mel_freq_bins), 
                            device=device, dtype=windows[0].dtype)
    weight_sum = torch.zeros_like(output_mel)
    
    overlap_mel_size = int(overlap_size * mel_scale_factor)
    
    for i, (window_mel, (start, end, actual_size)) in enumerate(zip(windows, window_positions)):
        mel_start = int(start * mel_scale_factor)
        mel_end = int(end * mel_scale_factor)
        actual_mel_size = window_mel.shape[2]
        
        weight_mask = torch.ones((1, 1, actual_mel_size, 1), device=device)
        
        if i > 0 and overlap_mel_size > 0:
            fade_in_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_in = torch.linspace(0, 1, fade_in_size, device=device)
            weight_mask[:, :, :fade_in_size, :] = fade_in.view(1, 1, -1, 1)
        
        if i < len(windows) - 1 and overlap_mel_size > 0:
            fade_out_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_out = torch.linspace(1, 0, fade_out_size, device=device)
            weight_mask[:, :, -fade_out_size:, :] = fade_out.view(1, 1, -1, 1)
        
        mel_end_actual = mel_start + actual_mel_size
        output_mel[:, :, mel_start:mel_end_actual, :] += window_mel * weight_mask
        weight_sum[:, :, mel_start:mel_end_actual, :] += weight_mask
    
    output_mel = output_mel / torch.clamp(weight_sum, min=1e-8)
    return output_mel

def preprocess_audio_for_panns(audio, target_sr=32000, chunk_duration=10.0):
    """
    Preprocess audio for PANNs model (resample to 32kHz, chunk into 10s segments)
    
    Args:
        audio: Audio tensor or array
        target_sr: Target sample rate (32kHz for PANNs)
        chunk_duration: Duration of each chunk in seconds
    
    Returns:
        processed_chunks: List of audio chunks ready for PANNs
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize audio to [-1, 1]
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    # Resample to 32kHz for PANNs if needed
    if target_sr != 16000:  # AudioLDM outputs at 16kHz
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / 16000))
    
    # Split into chunks
    chunk_samples = int(chunk_duration * target_sr)
    chunks = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) >= chunk_samples // 2:  # Only keep chunks that are at least half the target length
            # Pad to exact length if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
    
    return chunks

def extract_panns_features(panns_model, audio_chunks):
    """
    Extract PANNs features from audio chunks
    
    Args:
        panns_model: PANNs AudioTagging model
        audio_chunks: List of audio chunks (32kHz, 10s each)
    
    Returns:
        embeddings: Averaged embeddings (2048,)
        probabilities: Averaged class probabilities (527,)
    """
    all_embeddings = []
    all_probabilities = []
    
    for chunk in audio_chunks:
        # Extract features using PANNs (handle dict or tuple outputs across versions)
        out = panns_model.inference(chunk)
        if isinstance(out, dict):
            clipwise_output = out.get('clipwise_output')
            embedding = out.get('embedding')
        else:
            clipwise_output, embedding = out
        
        all_embeddings.append(embedding)
        all_probabilities.append(clipwise_output)
    
    # Average across chunks
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_probability = np.mean(all_probabilities, axis=0)
    
    return avg_embedding, avg_probability

def compute_clap_score(clap_model, text, audio_path):
    """
    Compute CLAP similarity score between text and audio
    
    Args:
        clap_model: LAION CLAP model
        text: Text prompt
        audio_path: Path to audio file
    
    Returns:
        clap_score: Cosine similarity score
    """
    try:
        # Get text embedding
        text_embedding = clap_model.get_text_embedding([text], use_tensor=True)
        
        # Get audio embedding  
        audio_embedding = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            text_embedding, audio_embedding, dim=1
        ).item()
        
        return similarity
    except Exception as e:
        print(f"    ⚠️  CLAP computation failed: {e}")
        return 0.0

def compute_fd_score(ref_embeddings, gen_embeddings):
    """
    Compute Fréchet Distance between reference and generated embeddings
    """
    # Convert to numpy if needed
    if isinstance(ref_embeddings, torch.Tensor):
        ref_embeddings = ref_embeddings.cpu().numpy()
    if isinstance(gen_embeddings, torch.Tensor):
        gen_embeddings = gen_embeddings.cpu().numpy()
    
    # Compute means and covariances
    mu_ref = np.mean(ref_embeddings, axis=0)
    mu_gen = np.mean(gen_embeddings, axis=0)
    
    sigma_ref = np.cov(ref_embeddings, rowvar=False)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)
    
    # Add small epsilon for numerical stability
    epsilon = 1e-10
    sigma_ref += epsilon * np.eye(sigma_ref.shape[0])
    sigma_gen += epsilon * np.eye(sigma_gen.shape[0])
    
    # Compute FD
    diff = mu_ref - mu_gen
    
    # Compute sqrt of product of covariances
    from scipy.linalg import sqrtm
    sqrt_sigma_product = sqrtm(sigma_ref @ sigma_gen)
    
    # Handle complex numbers from sqrtm
    if np.iscomplexobj(sqrt_sigma_product):
        sqrt_sigma_product = sqrt_sigma_product.real
    
    fd = np.sum(diff**2) + np.trace(sigma_ref) + np.trace(sigma_gen) - 2 * np.trace(sqrt_sigma_product)
    
    return max(0.0, fd)  # Ensure non-negative

def compute_is_score(probabilities):
    """
    Compute Inception Score from class probabilities
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    # Add epsilon for numerical stability
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    
    # Compute marginal probability
    p_y = np.mean(probabilities, axis=0)
    
    # Compute KL divergence for each sample
    kl_divs = []
    for p_yx in probabilities:
        kl_div = np.sum(p_yx * np.log(p_yx / p_y + epsilon))
        kl_divs.append(kl_div)
    
    # Inception Score is exp(mean(KL))
    is_score = np.exp(np.mean(kl_divs))
    
    return is_score

def compute_kl_score(ref_probs, gen_probs):
    """
    Compute KL divergence between reference and generated probability distributions
    """
    if isinstance(ref_probs, torch.Tensor):
        ref_probs = ref_probs.cpu().numpy()
    if isinstance(gen_probs, torch.Tensor):
        gen_probs = gen_probs.cpu().numpy()
    
    # Average across samples to get distributions
    p_ref = np.mean(ref_probs, axis=0)
    p_gen = np.mean(gen_probs, axis=0)
    
    # Add epsilon and normalize
    epsilon = 1e-10
    p_ref = p_ref + epsilon
    p_gen = p_gen + epsilon
    p_ref = p_ref / np.sum(p_ref)
    p_gen = p_gen / np.sum(p_gen)
    
    # Compute KL divergence
    kl = np.sum(p_ref * np.log(p_ref / p_gen))
    
    return kl

def evaluate_config(config_id, overlap_percent, overlap_size, val_samples, models, args):
    """
    Evaluate a single hyperparameter configuration
    
    Args:
        config_id: Configuration identifier
        overlap_percent: Overlap percentage parameter
        overlap_size: Overlap size parameter  
        val_samples: Validation samples
        models: Tuple of (audioldm_model, clap_model, panns_model)
        args: Command line arguments
    
    Returns:
        metrics: Dictionary of computed metrics
        per_sample_data: List of per-sample detailed results
    """
    audioldm_model, clap_model, panns_model = models
    
    print(f"    📊 Config {config_id}: overlap_percent={overlap_percent}, overlap_size={overlap_size}")
    
    # Create output directory for this config
    config_dir = f"val_gens/config_{config_id}"
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate audio for each validation sample
    clap_scores = []
    ref_embeddings = []
    gen_embeddings = []
    ref_probabilities = []
    gen_probabilities = []
    per_sample_data = []
    
    for i, sample in enumerate(tqdm(val_samples, desc=f"Config {config_id}")):
        prompt = sample['caption']
        sample_id = sample['id']
        ref_audio_path = sample['audio_path']
        
        # Initialize sample data record
        sample_data = {
            'config_id': config_id,
            'overlap_percent': overlap_percent,
            'overlap_size': overlap_size,
            'sample_id': sample_id,
            'prompt': prompt,
            'ref_audio_path': ref_audio_path,
            'gen_audio_path': '',
            'generation_success': False,
            'generation_error': '',
            'clap_score': 0.0,
            'ref_embedding_mean': 0.0,
            'ref_embedding_std': 0.0,
            'gen_embedding_mean': 0.0,
            'gen_embedding_std': 0.0,
            'ref_prob_entropy': 0.0,
            'gen_prob_entropy': 0.0,
            'ref_prob_max': 0.0,
            'gen_prob_max': 0.0,
            'audio_duration_actual': 0.0,
            'audio_samples_count': 0
        }
        
        # Generate audio
        try:
            start_time = time.time()
            gen_waveform = generate_audio_multidiffusion(
                model=audioldm_model,
                prompt=prompt,
                duration=args.duration,
                overlap_percent=overlap_percent,
                overlap_size=overlap_size
            )
            generation_time = time.time() - start_time
            
            # Save generated audio
            gen_audio_path = os.path.join(config_dir, f"wav{sample_id}.wav")
            import soundfile as sf
            sf.write(gen_audio_path, gen_waveform, 16000)
            
            # Update sample data
            sample_data['gen_audio_path'] = gen_audio_path
            sample_data['generation_success'] = True
            sample_data['audio_duration_actual'] = len(gen_waveform) / 16000
            sample_data['audio_samples_count'] = len(gen_waveform)
            sample_data['generation_time_seconds'] = generation_time
            
            # Compute CLAP score
            clap_score = compute_clap_score(clap_model, prompt, gen_audio_path)
            clap_scores.append(clap_score)
            sample_data['clap_score'] = clap_score
            
            # Extract PANNs features for reference audio
            if os.path.exists(ref_audio_path):
                ref_audio, _ = torchaudio.load(ref_audio_path)
                ref_audio = ref_audio.squeeze().numpy()
                ref_chunks = preprocess_audio_for_panns(ref_audio)
                ref_emb, ref_prob = extract_panns_features(panns_model, ref_chunks)
                ref_embeddings.append(ref_emb)
                ref_probabilities.append(ref_prob)
                
                # Store detailed ref statistics
                sample_data['ref_embedding_mean'] = float(np.mean(ref_emb))
                sample_data['ref_embedding_std'] = float(np.std(ref_emb))
                sample_data['ref_prob_entropy'] = float(-np.sum(ref_prob * np.log(ref_prob + 1e-10)))
                sample_data['ref_prob_max'] = float(np.max(ref_prob))
                sample_data['ref_audio_duration'] = len(ref_audio) / 16000 if hasattr(ref_audio, '__len__') else 0.0
            
            # Extract PANNs features for generated audio
            gen_chunks = preprocess_audio_for_panns(gen_waveform)
            gen_emb, gen_prob = extract_panns_features(panns_model, gen_chunks)
            gen_embeddings.append(gen_emb)
            gen_probabilities.append(gen_prob)
            
            # Store detailed gen statistics
            sample_data['gen_embedding_mean'] = float(np.mean(gen_emb))
            sample_data['gen_embedding_std'] = float(np.std(gen_emb))
            sample_data['gen_prob_entropy'] = float(-np.sum(gen_prob * np.log(gen_prob + 1e-10)))
            sample_data['gen_prob_max'] = float(np.max(gen_prob))
            
        except Exception as e:
            print(f"      ⚠️  Error processing sample {sample_id}: {e}")
            sample_data['generation_error'] = str(e)
            continue
        
        per_sample_data.append(sample_data)
    
    # Compute aggregate metrics
    metrics = {}
    
    # CLAP score (higher is better)
    metrics['clap'] = np.mean(clap_scores) if clap_scores else 0.0
    
    # FD score (lower is better)
    if ref_embeddings and gen_embeddings:
        metrics['fd'] = compute_fd_score(np.array(ref_embeddings), np.array(gen_embeddings))
    else:
        metrics['fd'] = float('inf')
    
    # IS score (higher is better)
    if gen_probabilities:
        metrics['is'] = compute_is_score(np.array(gen_probabilities))
    else:
        metrics['is'] = 0.0
    
    # KL score (lower is better)
    if ref_probabilities and gen_probabilities:
        metrics['kl'] = compute_kl_score(np.array(ref_probabilities), np.array(gen_probabilities))
    else:
        metrics['kl'] = float('inf')
    
    # Composite score (normalized sum: CLAP + IS - FD - KL)
    # Normalize components to [0, 1] range for fair combination
    clap_norm = max(0, min(1, metrics['clap']))  # CLAP is typically [-1, 1], map to [0, 1]
    is_norm = max(0, min(1, metrics['is'] / 10))  # IS is typically [1, 10], normalize
    fd_norm = max(0, min(1, 1 / (1 + metrics['fd'])))  # FD: lower is better, invert
    kl_norm = max(0, min(1, 1 / (1 + metrics['kl'])))  # KL: lower is better, invert
    
    metrics['composite'] = (clap_norm + is_norm + fd_norm + kl_norm) / 4
    
    print(f"      📈 CLAP: {metrics['clap']:.4f}, FD: {metrics['fd']:.4f}, IS: {metrics['is']:.4f}, KL: {metrics['kl']:.4f}")
    print(f"      🎯 Composite: {metrics['composite']:.4f}")
    
    return metrics, per_sample_data

def main():
    """Main tuning function"""
    args = parse_args()
    
    print(f"🎛️  HYPERPARAMETER GRID SEARCH")
    print(f"   Overlap Percents: {args.overlap_percents}")
    print(f"   Overlap Sizes: {args.overlap_sizes}")
    print(f"   Duration: {args.duration}s")
    print(f"   Validation Size: {args.val_size}")
    print()
    
    # Setup environment
    setup_environment(args.seed)
    
    # Load and split data
    data_path = 'AudioSet/data.json'
    audio_dir = 'AudioSet/downloaded_audio'
    val_samples, test_samples, splits = load_and_split_data(
        data_path, audio_dir, args.val_size, args.test_size, args.seed
    )
    
    # Initialize models
    models = initialize_models()
    
    # Grid search over hyperparameter combinations
    print(f"🔍 STARTING GRID SEARCH")
    print(f"   Total combinations: {len(args.overlap_percents) * len(args.overlap_sizes)}")
    print()
    
    all_results = []
    all_sample_data = []
    config_id = 0
    
    for overlap_percent, overlap_size in itertools.product(args.overlap_percents, args.overlap_sizes):
        config_id += 1
        
        print(f"  ⚙️  Evaluating Config {config_id}/{len(args.overlap_percents) * len(args.overlap_sizes)}")
        
        # Evaluate this configuration
        metrics, per_sample_data = evaluate_config(
            config_id=config_id,
            overlap_percent=overlap_percent,
            overlap_size=overlap_size,
            val_samples=val_samples,
            models=models,
            args=args
        )
        
        # Store results
        result = {
            'config_id': config_id,
            'overlap_percent': overlap_percent,
            'overlap_size': overlap_size,
            **metrics
        }
        all_results.append(result)
        all_sample_data.extend(per_sample_data)
        
        print()
    
    # Find best configuration
    best_config = max(all_results, key=lambda x: x['composite'])
    
    print(f"🏆 BEST CONFIGURATION")
    print(f"   Config ID: {best_config['config_id']}")
    print(f"   Overlap Percent: {best_config['overlap_percent']}")
    print(f"   Overlap Size: {best_config['overlap_size']}")
    print(f"   Composite Score: {best_config['composite']:.4f}")
    print(f"   CLAP: {best_config['clap']:.4f}")
    print(f"   FD: {best_config['fd']:.4f}")
    print(f"   IS: {best_config['is']:.4f}")
    print(f"   KL: {best_config['kl']:.4f}")
    print()
    
    # Save results
    results_data = {
        'args': vars(args),
        'splits': splits,
        'all_results': all_results,
        'best_config': best_config
    }
    
    with open('tuning_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save detailed CSV with all sample-level data
    import csv
    with open('tuning_detailed_results.csv', 'w', newline='') as csvfile:
        if all_sample_data:
            fieldnames = list(all_sample_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_sample_data:
                writer.writerow(row)
    
    # Save config summary CSV
    with open('tuning_config_summary.csv', 'w', newline='') as csvfile:
        if all_results:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
    
    print(f"💾 Results saved to:")
    print(f"   - tuning_results.json (complete results)")
    print(f"   - tuning_detailed_results.csv (per-sample detailed data)")
    print(f"   - tuning_config_summary.csv (per-config aggregate metrics)")
    print(f"✅ Hyperparameter tuning completed!")

if __name__ == '__main__':
    main()
