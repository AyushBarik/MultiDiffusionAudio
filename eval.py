#!/usr/bin/env python3
"""
Evaluation Script for AudioLDM MultiDiffusion vs Naive Chunking

This script evaluates the novel MultiDiffusion method against a naive chunking baseline
on test samples using optimal hyperparameters from tuning.

Usage:
    python eval.py [--optimal_config_path tuning_results.json]

Features:
- Compares MultiDiffusion (novel) vs Naive Chunking (baseline)
- Uses optimal hyperparameters from tuning
- Comprehensive metrics: CLAP, FD, IS, KL, Gemini evaluation
- Saves detailed results for analysis
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import laion_clap
import google.generativeai as genai

# Add AudioLDM path
sys.path.append('AudioLDM')
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size

# Add utils path  
sys.path.append('AudioLDM/utils')
from utils import *

# Ensure PANNs inference is available
try:
    from panns_inference import AudioTagging
except ImportError:
    print("Installing panns-inference...")
    os.system("pip install panns-inference")
    from panns_inference import AudioTagging

print("🔬 AUDIOLDM EVALUATION: MULTIDIFFUSION VS NAIVE CHUNKING")
print("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AudioLDM MultiDiffusion vs Naive Chunking Evaluation')
    parser.add_argument('--optimal_config_path', type=str, default='tuning_results.json',
                        help='Path to tuning results with optimal configuration')
    parser.add_argument('--duration', type=float, default=100.0,
                        help='Duration of generated audio in seconds')
    parser.add_argument('--gemini_api_key', type=str, default="AIzaSyCVX9GFTV9Mz4YTqd5IjymS49LE5Llyb0M",
                        help='Google Gemini API key for model-based evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_optimal_config(config_path):
    """Load optimal configuration from tuning results"""
    if not os.path.exists(config_path):
        print(f"⚠️  Optimal config file not found: {config_path}")
        print("   Using default parameters: overlap_percent=0.75, overlap_size=192")
        return {
            'overlap_percent': 0.75,
            'overlap_size': 192,
            'chunk_frames': 256
        }
    
    with open(config_path, 'r') as f:
        tuning_results = json.load(f)
    
    best_config = tuning_results['best_config']
    optimal_config = {
        'overlap_percent': best_config['overlap_percent'],
        'overlap_size': best_config['overlap_size'],
        'chunk_frames': 256  # Standard chunk size (matching naive chunking)
    }
    
    print(f"📂 Loaded optimal config from {config_path}")
    print(f"   Overlap Percent: {optimal_config['overlap_percent']}")
    print(f"   Overlap Size: {optimal_config['overlap_size']}")
    
    return optimal_config

def load_test_data():
    """Load test samples from splits"""
    splits_path = 'splits.json'
    if not os.path.exists(splits_path):
        print(f"❌ Splits file not found: {splits_path}")
        print("   Please run tuning.py first to create data splits")
        return []
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    # Load full dataset
    with open('AudioSet/data.json', 'r') as f:
        data = json.load(f)
    
    # Get test samples
    test_samples = []
    for sample in data['data']:
        if sample['id'] in splits['test_ids']:
            audio_path = f"AudioSet/downloaded_audio/wav{sample['id']}.wav"
            if os.path.exists(audio_path):
                test_samples.append({
                    'id': sample['id'],
                    'caption': sample['caption'],
                    'duration': sample['duration'],
                    'audio_path': audio_path
                })
    
    print(f"📂 Loaded {len(test_samples)} test samples")
    return test_samples

def initialize_models():
    """Initialize all required models"""
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
    panns_model = AudioTagging(checkpoint_path=None, model_type='Cnn14', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ All models initialized")
    return audioldm_model, clap_model, panns_model

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

def generate_audio_multidiffusion(model, prompt, duration, overlap_percent, overlap_size, 
                                  chunk_frames=256, ddim_steps=200):
    """Generate audio using MultiDiffusion method"""
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
        actual_overlap = max(calculated_overlap, overlap_size)
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

def generate_audio_naive_chunking(model, prompt, duration, chunk_frames=256, ddim_steps=200):
    """Generate audio using Naive Chunking method (no overlap)"""
    # Prepare conditioning
    text_emb = model.get_learned_conditioning([prompt])
    unconditional_embedding = model.get_learned_conditioning([""])
    
    # Setup parameters
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    z = torch.randn(shape).to(model.device)
    
    # Process chunks without overlap
    chunks = torch.split(z, chunk_frames, dim=2)
    processed_chunks = []
    
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            processed_chunk = multidiffusion_sample_clean(
                sampler=sampler,
                shape=chunk.shape,
                conditioning=text_emb,
                unconditional_conditioning=unconditional_embedding,
                unconditional_guidance_scale=3.0,
                eta=0.1,
                x_T=chunk,
                S=ddim_steps,
                chunk_frames=chunk_frames,
                overlap_frames=0  # No overlap for naive chunking
            )
            processed_chunks.append(processed_chunk)
        
        # Concatenate processed chunks
        samples = torch.cat(processed_chunks, dim=2)
        
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

def preprocess_audio_for_panns(audio, target_sr=32000, chunk_duration=10.0):
    """Preprocess audio for PANNs model"""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize audio to [-1, 1]
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    # Resample to 32kHz for PANNs if needed
    if target_sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / 16000))
    
    # Split into chunks
    chunk_samples = int(chunk_duration * target_sr)
    chunks = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) >= chunk_samples // 2:
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
    
    return chunks

def extract_panns_features(panns_model, audio_chunks):
    """Extract PANNs features from audio chunks"""
    all_embeddings = []
    all_probabilities = []
    
    for chunk in audio_chunks:
        clipwise_output, embedding = panns_model.inference(chunk)
        all_embeddings.append(embedding)
        all_probabilities.append(clipwise_output)
    
    # Average across chunks
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_probability = np.mean(all_probabilities, axis=0)
    
    return avg_embedding, avg_probability

def compute_clap_score(clap_model, text, audio_path):
    """Compute CLAP similarity score between text and audio"""
    try:
        text_embedding = clap_model.get_text_embedding([text], use_tensor=True)
        audio_embedding = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            text_embedding, audio_embedding, dim=1
        ).item()
        return similarity
    except Exception as e:
        print(f"    ⚠️  CLAP computation failed: {e}")
        return 0.0

def compute_fd_score(ref_embeddings, gen_embeddings):
    """Compute Fréchet Distance between reference and generated embeddings"""
    if isinstance(ref_embeddings, torch.Tensor):
        ref_embeddings = ref_embeddings.cpu().numpy()
    if isinstance(gen_embeddings, torch.Tensor):
        gen_embeddings = gen_embeddings.cpu().numpy()
    
    mu_ref = np.mean(ref_embeddings, axis=0)
    mu_gen = np.mean(gen_embeddings, axis=0)
    
    sigma_ref = np.cov(ref_embeddings, rowvar=False)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)
    
    epsilon = 1e-10
    sigma_ref += epsilon * np.eye(sigma_ref.shape[0])
    sigma_gen += epsilon * np.eye(sigma_gen.shape[0])
    
    diff = mu_ref - mu_gen
    
    from scipy.linalg import sqrtm
    sqrt_sigma_product = sqrtm(sigma_ref @ sigma_gen)
    
    if np.iscomplexobj(sqrt_sigma_product):
        sqrt_sigma_product = sqrt_sigma_product.real
    
    fd = np.sum(diff**2) + np.trace(sigma_ref) + np.trace(sigma_gen) - 2 * np.trace(sqrt_sigma_product)
    
    return max(0.0, fd)

def compute_is_score(probabilities):
    """Compute Inception Score from class probabilities"""
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    
    p_y = np.mean(probabilities, axis=0)
    
    kl_divs = []
    for p_yx in probabilities:
        kl_div = np.sum(p_yx * np.log(p_yx / p_y + epsilon))
        kl_divs.append(kl_div)
    
    is_score = np.exp(np.mean(kl_divs))
    return is_score

def compute_kl_score(ref_probs, gen_probs):
    """Compute KL divergence between reference and generated probability distributions"""
    if isinstance(ref_probs, torch.Tensor):
        ref_probs = ref_probs.cpu().numpy()
    if isinstance(gen_probs, torch.Tensor):
        gen_probs = gen_probs.cpu().numpy()
    
    p_ref = np.mean(ref_probs, axis=0)
    p_gen = np.mean(gen_probs, axis=0)
    
    epsilon = 1e-10
    p_ref = p_ref + epsilon
    p_gen = p_gen + epsilon
    p_ref = p_ref / np.sum(p_ref)
    p_gen = p_gen / np.sum(p_gen)
    
    kl = np.sum(p_ref * np.log(p_ref / p_gen))
    return kl

def evaluate_with_gemini(prompt, audio_path_a, audio_path_b, api_key):
    """
    Evaluate audio quality using Gemini model
    
    Args:
        prompt: Text prompt used for generation
        audio_path_a: Path to first audio (novel method)
        audio_path_b: Path to second audio (baseline method)  
        api_key: Google Gemini API key
    
    Returns:
        score: 7-point Likert scale score (-3 to 3)
        rationale: Explanation of the score
    """
    try:
        genai.configure(api_key=api_key)
        
        # Upload audio files
        audio_file_a = genai.upload_file(path=audio_path_a)
        audio_file_b = genai.upload_file(path=audio_path_b)
        
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        
        evaluation_prompt = [
            f"""
Please act as an impartial judge and evaluate the overall audio quality of the responses provided by two AI assistants. You should choose the assistant that produced the better audio.

Your evaluation should focus only on technical audio quality. Consider factors such as fidelity (is the audio clean and clear?), realism, unwanted glitches, noise, or poor transitions. 

The original text prompt was: "{prompt}"

You should start with your evaluation by comparing the two responses and provide a short rationale. After providing your rationale, you should output the final verdict by strictly following this seven-point Likert scale: 3 if assistant A is much better, 2 if assistant A is better, 1 if assistant A is slightly better, 0 if the two responses have roughly the same quality, -1 if assistant B is slightly better, -2 if assistant B is better, and -3 if assistant B is much better.

You should format as follows:

[Rationale]: 
[Score]:  """,
            "Audio File A (Novel Method):",
            audio_file_a,
            "Audio File B (Baseline Method):",
            audio_file_b,
        ]
        
        response = model.generate_content(evaluation_prompt)
        response_text = response.text
        
        # Parse the response to extract score and rationale
        lines = response_text.strip().split('\n')
        rationale = ""
        score = 0
        
        for line in lines:
            if line.startswith('[Rationale]:'):
                rationale = line.replace('[Rationale]:', '').strip()
            elif line.startswith('[Score]:'):
                score_text = line.replace('[Score]:', '').strip()
                try:
                    score = int(score_text)
                except ValueError:
                    score = 0
        
        return score, rationale
        
    except Exception as e:
        print(f"    ⚠️  Gemini evaluation failed: {e}")
        return 0, f"Error: {e}"

def generate_and_evaluate_samples(test_samples, models, optimal_config, args):
    """Generate audio samples and compute all metrics"""
    audioldm_model, clap_model, panns_model = models
    
    # Create output directories
    os.makedirs('test_gens/novel', exist_ok=True)
    os.makedirs('test_gens/baseline', exist_ok=True)
    
    # Storage for results
    per_sample_results = []
    
    # Storage for aggregate metrics
    ref_embeddings = []
    ref_probabilities = []
    novel_embeddings = []
    novel_probabilities = []
    baseline_embeddings = []
    baseline_probabilities = []
    
    novel_clap_scores = []
    baseline_clap_scores = []
    gemini_scores = []
    
    print(f"🎵 GENERATING AND EVALUATING {len(test_samples)} TEST SAMPLES")
    print()
    
    for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        sample_id = sample['id']
        prompt = sample['caption']
        ref_audio_path = sample['audio_path']
        
        print(f"  📝 Sample {i+1}/{len(test_samples)}: {prompt[:50]}...")
        
        # Initialize detailed sample record
        sample_result = {
            'sample_id': sample_id,
            'prompt': prompt,
            'ref_audio_path': ref_audio_path,
            'novel_audio_path': '',
            'baseline_audio_path': '',
            'novel_generation_success': False,
            'baseline_generation_success': False,
            'novel_generation_error': '',
            'baseline_generation_error': '',
            'novel_generation_time': 0.0,
            'baseline_generation_time': 0.0,
            'novel_audio_duration': 0.0,
            'baseline_audio_duration': 0.0,
            'novel_audio_samples': 0,
            'baseline_audio_samples': 0,
            'novel_clap': 0.0,
            'baseline_clap': 0.0,
            'clap_difference': 0.0,
            'gemini_score': 0,
            'gemini_rationale': '',
            'ref_embedding_mean': 0.0,
            'ref_embedding_std': 0.0,
            'novel_embedding_mean': 0.0,
            'novel_embedding_std': 0.0,
            'baseline_embedding_mean': 0.0,
            'baseline_embedding_std': 0.0,
            'ref_prob_entropy': 0.0,
            'novel_prob_entropy': 0.0,
            'baseline_prob_entropy': 0.0,
            'ref_prob_max': 0.0,
            'novel_prob_max': 0.0,
            'baseline_prob_max': 0.0,
            'embedding_distance_novel_ref': 0.0,
            'embedding_distance_baseline_ref': 0.0,
            'embedding_distance_novel_baseline': 0.0,
            'optimal_overlap_percent': optimal_config['overlap_percent'],
            'optimal_overlap_size': optimal_config['overlap_size'],
            'chunk_frames': optimal_config['chunk_frames']
        }
        
        try:
            # Generate novel method audio (MultiDiffusion)
            print(f"    🔬 Generating novel (MultiDiffusion)...")
            start_time = time.time()
            novel_waveform = generate_audio_multidiffusion(
                model=audioldm_model,
                prompt=prompt,
                duration=args.duration,
                overlap_percent=optimal_config['overlap_percent'],
                overlap_size=optimal_config['overlap_size'],
                chunk_frames=optimal_config['chunk_frames']
            )
            novel_generation_time = time.time() - start_time
            
            novel_path = f"test_gens/novel/wav{sample_id}.wav"
            import soundfile as sf
            sf.write(novel_path, novel_waveform, 16000)
            
            # Update sample record - novel
            sample_result['novel_audio_path'] = novel_path
            sample_result['novel_generation_success'] = True
            sample_result['novel_generation_time'] = novel_generation_time
            sample_result['novel_audio_duration'] = len(novel_waveform) / 16000
            sample_result['novel_audio_samples'] = len(novel_waveform)
            
            # Generate baseline method audio (Naive Chunking)
            print(f"    📏 Generating baseline (Naive Chunking)...")
            start_time = time.time()
            baseline_waveform = generate_audio_naive_chunking(
                model=audioldm_model,
                prompt=prompt,
                duration=args.duration,
                chunk_frames=optimal_config['chunk_frames']
            )
            baseline_generation_time = time.time() - start_time
            
            baseline_path = f"test_gens/baseline/wav{sample_id}.wav"
            sf.write(baseline_path, baseline_waveform, 16000)
            
            # Update sample record - baseline
            sample_result['baseline_audio_path'] = baseline_path
            sample_result['baseline_generation_success'] = True
            sample_result['baseline_generation_time'] = baseline_generation_time
            sample_result['baseline_audio_duration'] = len(baseline_waveform) / 16000
            sample_result['baseline_audio_samples'] = len(baseline_waveform)
            
            # Compute CLAP scores
            print(f"    📊 Computing CLAP scores...")
            novel_clap = compute_clap_score(clap_model, prompt, novel_path)
            baseline_clap = compute_clap_score(clap_model, prompt, baseline_path)
            
            sample_result['novel_clap'] = novel_clap
            sample_result['baseline_clap'] = baseline_clap
            sample_result['clap_difference'] = novel_clap - baseline_clap
            
            novel_clap_scores.append(novel_clap)
            baseline_clap_scores.append(baseline_clap)
            
            # Extract PANNs features for reference audio
            if os.path.exists(ref_audio_path):
                ref_audio, _ = torchaudio.load(ref_audio_path)
                ref_audio = ref_audio.squeeze().numpy()
                ref_chunks = preprocess_audio_for_panns(ref_audio)
                ref_emb, ref_prob = extract_panns_features(panns_model, ref_chunks)
                ref_embeddings.append(ref_emb)
                ref_probabilities.append(ref_prob)
                
                # Store reference statistics
                sample_result['ref_embedding_mean'] = float(np.mean(ref_emb))
                sample_result['ref_embedding_std'] = float(np.std(ref_emb))
                sample_result['ref_prob_entropy'] = float(-np.sum(ref_prob * np.log(ref_prob + 1e-10)))
                sample_result['ref_prob_max'] = float(np.max(ref_prob))
            
            # Extract PANNs features for novel method
            print(f"    🔍 Extracting PANNs features...")
            novel_chunks = preprocess_audio_for_panns(novel_waveform)
            novel_emb, novel_prob = extract_panns_features(panns_model, novel_chunks)
            novel_embeddings.append(novel_emb)
            novel_probabilities.append(novel_prob)
            
            # Store novel statistics
            sample_result['novel_embedding_mean'] = float(np.mean(novel_emb))
            sample_result['novel_embedding_std'] = float(np.std(novel_emb))
            sample_result['novel_prob_entropy'] = float(-np.sum(novel_prob * np.log(novel_prob + 1e-10)))
            sample_result['novel_prob_max'] = float(np.max(novel_prob))
            
            # Extract PANNs features for baseline method
            baseline_chunks = preprocess_audio_for_panns(baseline_waveform)
            baseline_emb, baseline_prob = extract_panns_features(panns_model, baseline_chunks)
            baseline_embeddings.append(baseline_emb)
            baseline_probabilities.append(baseline_prob)
            
            # Store baseline statistics
            sample_result['baseline_embedding_mean'] = float(np.mean(baseline_emb))
            sample_result['baseline_embedding_std'] = float(np.std(baseline_emb))
            sample_result['baseline_prob_entropy'] = float(-np.sum(baseline_prob * np.log(baseline_prob + 1e-10)))
            sample_result['baseline_prob_max'] = float(np.max(baseline_prob))
            
            # Compute embedding distances
            if ref_embeddings:
                sample_result['embedding_distance_novel_ref'] = float(np.linalg.norm(novel_emb - ref_emb))
                sample_result['embedding_distance_baseline_ref'] = float(np.linalg.norm(baseline_emb - ref_emb))
            sample_result['embedding_distance_novel_baseline'] = float(np.linalg.norm(novel_emb - baseline_emb))
            
            # Gemini evaluation
            print(f"    🤖 Running Gemini evaluation...")
            gemini_score, gemini_rationale = evaluate_with_gemini(
                prompt, novel_path, baseline_path, args.gemini_api_key
            )
            gemini_scores.append(gemini_score)
            
            sample_result['gemini_score'] = gemini_score
            sample_result['gemini_rationale'] = gemini_rationale
            
            print(f"    ✅ CLAP Novel: {novel_clap:.4f}, Baseline: {baseline_clap:.4f}, Gemini: {gemini_score}")
            
        except Exception as e:
            print(f"    ❌ Error processing sample {sample_id}: {e}")
            sample_result['novel_generation_error'] = str(e) if 'novel_waveform' not in locals() else ''
            sample_result['baseline_generation_error'] = str(e) if 'baseline_waveform' not in locals() else ''
            continue
        
        # Store per-sample results
        per_sample_results.append(sample_result)
        
        print()
    
    # Compute aggregate metrics
    print(f"📈 COMPUTING AGGREGATE METRICS")
    
    aggregate_metrics = {}
    
    # CLAP scores
    aggregate_metrics['clap_novel_mean'] = np.mean(novel_clap_scores)
    aggregate_metrics['clap_baseline_mean'] = np.mean(baseline_clap_scores)
    
    # FD scores
    if ref_embeddings and novel_embeddings:
        aggregate_metrics['fd_novel'] = compute_fd_score(np.array(ref_embeddings), np.array(novel_embeddings))
    if ref_embeddings and baseline_embeddings:
        aggregate_metrics['fd_baseline'] = compute_fd_score(np.array(ref_embeddings), np.array(baseline_embeddings))
    
    # IS scores
    if novel_probabilities:
        aggregate_metrics['is_novel'] = compute_is_score(np.array(novel_probabilities))
    if baseline_probabilities:
        aggregate_metrics['is_baseline'] = compute_is_score(np.array(baseline_probabilities))
    
    # KL scores
    if ref_probabilities and novel_probabilities:
        aggregate_metrics['kl_novel'] = compute_kl_score(np.array(ref_probabilities), np.array(novel_probabilities))
    if ref_probabilities and baseline_probabilities:
        aggregate_metrics['kl_baseline'] = compute_kl_score(np.array(ref_probabilities), np.array(baseline_probabilities))
    
    # Gemini scores
    aggregate_metrics['gemini_mean'] = np.mean(gemini_scores)
    aggregate_metrics['gemini_novel_wins'] = sum(1 for score in gemini_scores if score > 0)
    aggregate_metrics['gemini_baseline_wins'] = sum(1 for score in gemini_scores if score < 0)
    aggregate_metrics['gemini_ties'] = sum(1 for score in gemini_scores if score == 0)
    
    return per_sample_results, aggregate_metrics

def main():
    """Main evaluation function"""
    args = parse_args()
    
    print(f"🎯 EVALUATION CONFIGURATION")
    print(f"   Duration: {args.duration}s")
    print(f"   Optimal Config: {args.optimal_config_path}")
    print()
    
    # Setup environment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load optimal configuration
    optimal_config = load_optimal_config(args.optimal_config_path)
    
    # Load test data
    test_samples = load_test_data()
    if not test_samples:
        return
    
    # Initialize models
    models = initialize_models()
    
    # Generate and evaluate samples
    per_sample_results, aggregate_metrics = generate_and_evaluate_samples(
        test_samples, models, optimal_config, args
    )
    
    # Print results
    print(f"🏆 FINAL EVALUATION RESULTS")
    print(f"=" * 50)
    print(f"📊 CLAP Scores:")
    print(f"   Novel (MultiDiffusion): {aggregate_metrics['clap_novel_mean']:.4f}")
    print(f"   Baseline (Naive): {aggregate_metrics['clap_baseline_mean']:.4f}")
    print(f"   Improvement: {aggregate_metrics['clap_novel_mean'] - aggregate_metrics['clap_baseline_mean']:+.4f}")
    print()
    
    print(f"📊 FD Scores (lower is better):")
    print(f"   Novel: {aggregate_metrics.get('fd_novel', 'N/A'):.4f if 'fd_novel' in aggregate_metrics else 'N/A'}")
    print(f"   Baseline: {aggregate_metrics.get('fd_baseline', 'N/A'):.4f if 'fd_baseline' in aggregate_metrics else 'N/A'}")
    print()
    
    print(f"📊 IS Scores (higher is better):")
    print(f"   Novel: {aggregate_metrics.get('is_novel', 'N/A'):.4f if 'is_novel' in aggregate_metrics else 'N/A'}")
    print(f"   Baseline: {aggregate_metrics.get('is_baseline', 'N/A'):.4f if 'is_baseline' in aggregate_metrics else 'N/A'}")
    print()
    
    print(f"📊 KL Scores (lower is better):")
    print(f"   Novel: {aggregate_metrics.get('kl_novel', 'N/A'):.4f if 'kl_novel' in aggregate_metrics else 'N/A'}")
    print(f"   Baseline: {aggregate_metrics.get('kl_baseline', 'N/A'):.4f if 'kl_baseline' in aggregate_metrics else 'N/A'}")
    print()
    
    print(f"🤖 Gemini Evaluation:")
    print(f"   Average Score: {aggregate_metrics['gemini_mean']:.2f}")
    print(f"   Novel Wins: {aggregate_metrics['gemini_novel_wins']}")
    print(f"   Baseline Wins: {aggregate_metrics['gemini_baseline_wins']}")
    print(f"   Ties: {aggregate_metrics['gemini_ties']}")
    print()
    
    # Save results
    results_data = {
        'args': vars(args),
        'optimal_config': optimal_config,
        'per_sample_results': per_sample_results,
        'aggregate_metrics': aggregate_metrics,
        'summary': {
            'total_samples': len(per_sample_results),
            'novel_method': 'MultiDiffusion',
            'baseline_method': 'Naive Chunking'
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save comprehensive detailed CSV with all available data
    import csv
    with open('evaluation_detailed_results.csv', 'w', newline='') as csvfile:
        if per_sample_results:
            fieldnames = list(per_sample_results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in per_sample_results:
                writer.writerow(result)
    
    # Save aggregate metrics CSV
    aggregate_list = []
    for key, value in aggregate_metrics.items():
        aggregate_list.append({
            'metric_name': key,
            'value': value,
            'metric_type': 'aggregate'
        })
    
    with open('evaluation_aggregate_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['metric_name', 'value', 'metric_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_list:
            writer.writerow(row)
    
    # Save pairwise comparison CSV (novel vs baseline per sample)
    comparison_data = []
    for result in per_sample_results:
        if result['novel_generation_success'] and result['baseline_generation_success']:
            comparison_data.append({
                'sample_id': result['sample_id'],
                'prompt': result['prompt'],
                'novel_clap': result['novel_clap'],
                'baseline_clap': result['baseline_clap'],
                'clap_difference': result['clap_difference'],
                'clap_improvement_percent': (result['clap_difference'] / result['baseline_clap'] * 100) if result['baseline_clap'] != 0 else 0,
                'novel_better_clap': result['novel_clap'] > result['baseline_clap'],
                'gemini_score': result['gemini_score'],
                'novel_wins_gemini': result['gemini_score'] > 0,
                'novel_generation_time': result['novel_generation_time'],
                'baseline_generation_time': result['baseline_generation_time'],
                'time_difference': result['novel_generation_time'] - result['baseline_generation_time'],
                'embedding_distance_novel_baseline': result['embedding_distance_novel_baseline'],
                'novel_prob_entropy': result['novel_prob_entropy'],
                'baseline_prob_entropy': result['baseline_prob_entropy'],
                'entropy_difference': result['novel_prob_entropy'] - result['baseline_prob_entropy']
            })
    
    with open('evaluation_pairwise_comparison.csv', 'w', newline='') as csvfile:
        if comparison_data:
            fieldnames = list(comparison_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison_data:
                writer.writerow(row)
    
    print(f"💾 Results saved to:")
    print(f"   - evaluation_results.json (complete results)")
    print(f"   - evaluation_detailed_results.csv (per-sample comprehensive data)")
    print(f"   - evaluation_aggregate_metrics.csv (aggregate metrics)")
    print(f"   - evaluation_pairwise_comparison.csv (head-to-head comparisons)")
    print(f"✅ Evaluation completed!")

if __name__ == '__main__':
    main()
