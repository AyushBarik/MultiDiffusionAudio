#!/usr/bin/env python3
"""
Score validation audio and pick best hyperparameter configuration

Run this in benchenv (evaluation environment).
No AudioLDM imports - only evaluation libraries.
"""

import os
import json
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import csv
import scipy.signal
import time

# Evaluation libraries (benchenv only)
import laion_clap
from panns_inference import AudioTagging
import google.generativeai as genai

def parse_args():
    parser = argparse.ArgumentParser(description='Score validation audio for tuning')
    parser.add_argument('--val-root', required=True, help='Root directory with config_* subdirs')
    parser.add_argument('--splits', required=True, help='Path to splits.json')
    parser.add_argument('--out-root', required=True, help='Output directory for results')
    parser.add_argument('--gemini-api-key', required=True, help='Gemini API key for evaluation')
    return parser.parse_args()

def initialize_models():
    """Initialize evaluation models"""
    print("🤖 Initializing evaluation models...")
    
    # Initialize CLAP
    print("  - Loading CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    clap_ckpt = os.path.expanduser('~/.cache/torch/hub/checkpoints/music_audioset_epoch_15_esc_90.14.pt')
    if os.path.exists(clap_ckpt):
        clap_model.load_ckpt(ckpt=clap_ckpt)
        print(f"    ✅ Loaded CLAP checkpoint: {clap_ckpt}")
    
    # Initialize PANNs
    print("  - Loading PANNs model...")
    panns_model = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    return clap_model, panns_model

def compute_clap_score(clap_model, text, audio_path):
    """Compute CLAP similarity score"""
    try:
        text_embedding = clap_model.get_text_embedding([text], use_tensor=True)
        audio_embedding = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(text_embedding, audio_embedding, dim=1).item()
        return similarity
    except Exception as e:
        print(f"    ⚠️  CLAP computation failed: {e}")
        return 0.0

def preprocess_audio_for_panns(audio, target_sr=32000, chunk_duration=10.0):
    """Preprocess audio for PANNs - resample to 32kHz"""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    # Resample from 16kHz to 32kHz for PANNs
    if target_sr != 16000:
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

def resample_reference_audio(ref_path, target_sr=16000):
    """Load and resample reference audio to match generated audio sample rate"""
    try:
        ref_audio, orig_sr = torchaudio.load(ref_path)
        if orig_sr != target_sr:
            ref_audio = torchaudio.functional.resample(ref_audio, orig_sr, target_sr)
        
        # Handle different audio shapes robustly
        if ref_audio.dim() > 1:
            ref_audio = ref_audio.mean(dim=0)  # Convert stereo to mono by averaging
        ref_audio = ref_audio.squeeze().numpy()
        # Ensure we have a 1D array
        if ref_audio.ndim == 0:
            ref_audio = np.array([ref_audio])
        elif ref_audio.ndim > 1:
            ref_audio = ref_audio.flatten()
            
        return ref_audio
    except Exception as e:
        print(f"        ⚠️  Error loading reference audio {ref_path}: {e}")
        return None

def compute_gemini_score(ref_audio_path, gen_audio_path, prompt, api_key):
    """Compute Gemini evaluation score using your exact setup"""
    try:
        genai.configure(api_key=api_key)
        
        print(f"    📤 Uploading reference audio...")
        ref_file = genai.upload_file(path=ref_audio_path)
        
        print(f"    📤 Uploading generated audio...")  
        gen_file = genai.upload_file(path=gen_audio_path)
        
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        
        prompt_template = [
            """Please act as an impartial judge and evaluate the overall audio quality of the responses provided by two AI assistants. You should choose the assistant that produced the better audio.

Your evaluation should focus only on technical audio quality. Consider factors such as fidelity (is the audio clean and clear?), realism, unwanted glitches, noise, or poor transitions. 

You should start with your evaluation by comparing the two responses and provide a short rationale. After providing your rationale, you should output the final verdict by strictly following this seven-point Likert scale: 3 if assistant A is much better, 2 if assistant A is better, 1 if assistant A is slightly better, 0 if the two responses have roughly the same quality, -1 if assistant B is slightly better, -2 if assistant B is better, and -3 if assistant B is much better.

You should format as follows:

[Rationale]: 
[Score]:  """,
            "Audio File A (Reference):",
            ref_file,
            "Audio File B (Generated):",
            gen_file,
        ]
        
        response = model.generate_content(prompt_template)
        
        # Parse score from response
        response_text = response.text
        lines = response_text.split('\n')
        score = None
        for line in lines:
            if '[Score]:' in line:
                score_text = line.split('[Score]:')[-1].strip()
                try:
                    score = int(score_text)
                except:
                    score = 0
                break
        
        if score is None:
            score = 0
            
        return {
            'score': score,
            'rationale': response_text,
            'ref_file_id': ref_file.name,
            'gen_file_id': gen_file.name
        }
        
    except Exception as e:
        print(f"    ⚠️  Gemini evaluation failed: {e}")
        return {
            'score': 0,
            'rationale': f"Error: {e}",
            'ref_file_id': None,
            'gen_file_id': None
        }

def extract_panns_features(panns_model, audio_chunks):
    """Extract PANNs features"""
    all_embeddings = []
    all_probabilities = []
    
    for chunk in audio_chunks:
        out = panns_model.inference(chunk)
        if isinstance(out, dict):
            clipwise_output = out.get('clipwise_output')
            embedding = out.get('embedding')
        else:
            clipwise_output, embedding = out
        
        all_embeddings.append(embedding)
        all_probabilities.append(clipwise_output)
    
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_probability = np.mean(all_probabilities, axis=0)
    
    return avg_embedding, avg_probability

def compute_fd_score(ref_embeddings, gen_embeddings):
    """Compute Fréchet Distance"""
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
    """Compute Inception Score"""
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
    """Compute KL divergence"""
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

def evaluate_config(config_dir, config_info, val_samples, models, gemini_api_key):
    """Evaluate a single configuration"""
    clap_model, panns_model = models
    
    print(f"    📊 Evaluating {config_dir}")
    
    clap_scores = []
    ref_embeddings = []
    gen_embeddings = []
    ref_probabilities = []
    gen_probabilities = []
    per_sample_data = []
    gemini_results = []
    
    for sample in tqdm(val_samples, desc=f"Config {config_info['id']}"):
        sample_id = sample['id']
        prompt = sample['caption']
        ref_audio_path = sample['audio_path']
        gen_audio_path = os.path.join(config_dir, f"wav{sample_id}.wav")
        
        sample_data = {
            'config_id': config_info['id'],
            'overlap_percent': config_info['overlap_percent'],
            'chunk_frames': config_info['chunk_frames'],
            'sample_id': sample_id,
            'prompt': prompt,
            'ref_audio_path': ref_audio_path,
            'gen_audio_path': gen_audio_path,
            'generation_success': False,
            'clap_score': 0.0,
            'gemini_score': 0
        }
        
        if not os.path.exists(gen_audio_path):
            print(f"      ⚠️  Missing: {gen_audio_path}")
            per_sample_data.append(sample_data)
            continue
        
        try:
            # CLAP score (text-audio similarity)
            clap_score = compute_clap_score(clap_model, prompt, gen_audio_path)
            clap_scores.append(clap_score)
            sample_data['clap_score'] = clap_score
            sample_data['generation_success'] = True
            
            # Load and resample reference audio to 16kHz to match generated
            if os.path.exists(ref_audio_path):
                ref_audio = resample_reference_audio(ref_audio_path, target_sr=16000)
                if ref_audio is not None:
                    ref_chunks = preprocess_audio_for_panns(ref_audio)
                    ref_emb, ref_prob = extract_panns_features(panns_model, ref_chunks)
                    ref_embeddings.append(ref_emb)
                    ref_probabilities.append(ref_prob)
                else:
                    print(f"        ⚠️  Skipping reference audio processing for sample {sample_id}")
            else:
                print(f"        ⚠️  Reference audio not found: {ref_audio_path}")
            
            # Load generated audio (already 16kHz from AudioLDM)
            gen_audio, _ = torchaudio.load(gen_audio_path)
            # Handle different audio shapes robustly
            if gen_audio.dim() > 1:
                gen_audio = gen_audio.mean(dim=0)  # Convert stereo to mono by averaging
            gen_audio = gen_audio.squeeze().numpy()
            # Ensure we have a 1D array
            if gen_audio.ndim == 0:
                gen_audio = np.array([gen_audio])
            elif gen_audio.ndim > 1:
                gen_audio = gen_audio.flatten()
            gen_chunks = preprocess_audio_for_panns(gen_audio)
            gen_emb, gen_prob = extract_panns_features(panns_model, gen_chunks)
            gen_embeddings.append(gen_emb)
            gen_probabilities.append(gen_prob)
            
            # Gemini evaluation
            print(f"      🤖 Running Gemini evaluation for sample {sample_id}...")
            gemini_result = compute_gemini_score(ref_audio_path, gen_audio_path, prompt, gemini_api_key)
            sample_data['gemini_score'] = gemini_result['score']
            sample_data['gemini_rationale'] = gemini_result['rationale']
            gemini_results.append(gemini_result)
            
            # Add small delay to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"      ⚠️  Error processing {sample_id}: {e}")
        
        per_sample_data.append(sample_data)
    
    # Compute aggregate metrics
    metrics = {
        'config_id': config_info['id'],
        'overlap_percent': config_info['overlap_percent'],
        'chunk_frames': config_info['chunk_frames']
    }
    
    metrics['clap'] = np.mean(clap_scores) if clap_scores else 0.0
    metrics['gemini'] = np.mean([r['score'] for r in gemini_results]) if gemini_results else 0.0
    
    if ref_embeddings and gen_embeddings:
        metrics['fd'] = compute_fd_score(np.array(ref_embeddings), np.array(gen_embeddings))
    else:
        metrics['fd'] = float('inf')
    
    if gen_probabilities:
        metrics['is'] = compute_is_score(np.array(gen_probabilities))
    else:
        metrics['is'] = 0.0
    
    if ref_probabilities and gen_probabilities:
        metrics['kl'] = compute_kl_score(np.array(ref_probabilities), np.array(gen_probabilities))
    else:
        metrics['kl'] = float('inf')
    
    # Composite score
    clap_norm = max(0, min(1, (metrics['clap'] + 1) / 2))  # [-1,1] -> [0,1]
    gemini_norm = max(0, min(1, (metrics['gemini'] + 3) / 6))  # [-3,3] -> [0,1]
    is_norm = max(0, min(1, metrics['is'] / 10))  # [1,10] -> [0,1]
    fd_norm = max(0, min(1, 1 / (1 + metrics['fd'])))  # lower is better
    kl_norm = max(0, min(1, 1 / (1 + metrics['kl'])))  # lower is better
    
    metrics['composite'] = (clap_norm + gemini_norm + is_norm + fd_norm + kl_norm) / 5
    
    print(f"      📈 CLAP: {metrics['clap']:.4f}, Gemini: {metrics['gemini']:.2f}, FD: {metrics['fd']:.4f}, IS: {metrics['is']:.4f}, KL: {metrics['kl']:.4f}")
    print(f"      🎯 Composite: {metrics['composite']:.4f}")
    
    return metrics, per_sample_data, gemini_results

def main():
    args = parse_args()
    
    print("🔧 VALIDATION AUDIO SCORING")
    print("=" * 50)
    
    # Load splits
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    # Load dataset to get sample info
    with open('AudioSet/data.json', 'r') as f:
        dataset = json.load(f)
    
    data_by_id = {str(sample['id']): sample for sample in dataset['data']}
    
    # Build validation samples
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
    
    print(f"📂 Found {len(val_samples)} validation samples")
    
    # Initialize models
    models = initialize_models()
    
    # Find all config directories
    config_dirs = []
    for item in os.listdir(args.val_root):
        if item.startswith('config_'):
            config_id = int(item.split('_')[1])
            config_path = os.path.join(args.val_root, item)
            
            # Load config info
            config_json_path = os.path.join(config_path, 'config.json')
            if os.path.exists(config_json_path):
                with open(config_json_path, 'r') as f:
                    config_info = json.load(f)
                # Ensure compatibility - add 'id' field if missing
                if 'id' not in config_info and 'config_id' in config_info:
                    config_info['id'] = config_info['config_id']
                elif 'id' not in config_info:
                    config_info['id'] = config_id
            else:
                # Fallback for old format
                config_info = {
                    'id': config_id,
                    'overlap_percent': 0.5,  # Default
                    'chunk_frames': 128      # Default
                }
            
            config_dirs.append((config_id, config_path, config_info))
    
    config_dirs.sort()  # Sort by config ID
    print(f"🔍 Found {len(config_dirs)} configurations to evaluate")
    
    # Evaluate each configuration
    all_results = []
    all_sample_data = []
    all_gemini_data = []
    
    for config_id, config_dir, config_info in config_dirs:
        metrics, per_sample_data, gemini_results = evaluate_config(
            config_dir, config_info, val_samples, models, args.gemini_api_key
        )
        all_results.append(metrics)
        all_sample_data.extend(per_sample_data)
        all_gemini_data.extend(gemini_results)
    
    # Find best configuration
    if all_results:
        best_config = max(all_results, key=lambda x: x['composite'])
        
        print(f"\n🏆 BEST CONFIGURATION")
        print(f"   Config ID: {best_config['config_id']}")
        print(f"   Composite Score: {best_config['composite']:.4f}")
        print(f"   CLAP: {best_config['clap']:.4f}")
        print(f"   Gemini: {best_config['gemini']:.2f}")
        print(f"   FD: {best_config['fd']:.4f}")
        print(f"   IS: {best_config['is']:.4f}")
        print(f"   KL: {best_config['kl']:.4f}")
    else:
        best_config = None
        print("⚠️  No valid configurations found")
    
    # Create output directory
    os.makedirs(args.out_root, exist_ok=True)
    
    # Save results
    results_data = {
        'splits_file': args.splits,
        'val_root': args.val_root,
        'all_results': all_results,
        'best_config': best_config
    }
    
    results_path = os.path.join(args.out_root, 'tuning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save detailed CSV
    detailed_path = os.path.join(args.out_root, 'tuning_detailed_results.csv')
    if all_sample_data:
        with open(detailed_path, 'w', newline='') as csvfile:
            fieldnames = list(all_sample_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_sample_data:
                writer.writerow(row)
    
    # Save config summary CSV
    summary_path = os.path.join(args.out_root, 'tuning_config_summary.csv')
    if all_results:
        with open(summary_path, 'w', newline='') as csvfile:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
    
    # Save Gemini evaluation details
    gemini_path = os.path.join(args.out_root, 'tuning_gemini_results.csv')
    if all_gemini_data:
        with open(gemini_path, 'w', newline='') as csvfile:
            fieldnames = ['score', 'rationale', 'ref_file_id', 'gen_file_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_gemini_data:
                writer.writerow(row)
    
    print(f"\n💾 Results saved to:")
    print(f"   - {results_path}")
    print(f"   - {detailed_path}")
    print(f"   - {summary_path}")
    print(f"   - {gemini_path}")
    print(f"✅ Validation scoring completed!")

if __name__ == '__main__':
    main()
