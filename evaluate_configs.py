#!/usr/bin/env python3
"""
Proper evaluation: Compare each MultiDiffusion config against AudioSet reference audio
"""

import os
import json
import torch
import glob
import shutil
from audioldm_eval import EvaluationHelper

def chunk_audio_for_eval(input_dir, output_dir, chunk_length_sec=10, sr=16000):
    """Chunk audio files into 10-second segments for evaluation"""
    import torchaudio
    import soundfile as sf
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing chunks
    for file in glob.glob(os.path.join(output_dir, "*.wav")):
        os.remove(file)
    
    total_chunks = 0
    
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.wav'):
            print(f"  Chunking {file}...")
            
            # Load audio
            audio, orig_sr = torchaudio.load(os.path.join(input_dir, file))
            
            # Resample if needed
            if orig_sr != sr:
                audio = torchaudio.functional.resample(audio, orig_sr, sr)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(0)
            else:
                audio = audio.squeeze(0)
            
            # Split into chunks
            chunk_size = chunk_length_sec * sr
            file_chunks = 0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only keep full chunks
                    chunk_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_chunk{i//chunk_size:03d}.wav")
                    sf.write(chunk_path, chunk.numpy(), sr)
                    file_chunks += 1
                    total_chunks += 1
            
            print(f"    → {file_chunks} chunks")
    
    print(f"  Total: {total_chunks} chunks")
    return total_chunks

def evaluate_config(config_id, evaluator, reference_dir):
    """Evaluate a single config against reference"""
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING CONFIG {config_id}")
    print(f"{'='*60}")
    
    # Paths
    config_dir = f"artifacts/val/novel/config_{config_id}"
    config_file = os.path.join(config_dir, "config.json")
    
    # Check if config exists
    if not os.path.exists(config_dir):
        print(f"❌ Config {config_id} not found!")
        return None
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"📋 Configuration {config_id}:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create temporary directories for chunked audio
    temp_ref_dir = f"temp_ref_config_{config_id}"
    temp_gen_dir = f"temp_gen_config_{config_id}"
    
    try:
        # Chunk reference audio (AudioSet)
        print(f"\n📁 Chunking reference audio from {reference_dir}...")
        ref_chunks = chunk_audio_for_eval(reference_dir, temp_ref_dir)
        
        # Chunk generated audio (this config)
        print(f"\n📁 Chunking generated audio from {config_dir}...")
        gen_chunks = chunk_audio_for_eval(config_dir, temp_gen_dir)
        
        print(f"\n📊 Reference chunks: {ref_chunks}")
        print(f"📊 Generated chunks: {gen_chunks}")
        
        if ref_chunks == 0 or gen_chunks == 0:
            print("❌ No audio chunks found!")
            return None
        
        # Evaluate
        print(f"\n🚀 Running evaluation...")
        print("This may take several minutes...")
        
        results = evaluator.calculate_metrics(
            generate_files_path=temp_gen_dir,
            groundtruth_path=temp_ref_dir,
            same_name=True,
            limit_num=None,
            calculate_psnr_ssim=False,
            calculate_lsd=False,
            recalculate=True  # Force recalculation for each config
        )
        
        # Extract key metrics
        fad = results.get('frechet_audio_distance', 'N/A')
        is_mean = results.get('inception_score_mean', 'N/A')
        is_std = results.get('inception_score_std', 'N/A')
        kl_soft = results.get('kullback_leibler_divergence_softmax', 'N/A')
        kl_sig = results.get('kullback_leibler_divergence_sigmoid', 'N/A')
        
        # Results
        print(f"\n🎯 RESULTS FOR CONFIG {config_id}:")
        print(f"   FAD (Fréchet Audio Distance): {fad}")
        print(f"   IS (Inception Score): {is_mean} ± {is_std}")
        print(f"   KL Divergence (Softmax): {kl_soft}")
        print(f"   KL Divergence (Sigmoid): {kl_sig}")
        
        # Return structured results
        result_data = {
            'config_id': config_id,
            'config': config,
            'metrics': {
                'fad': fad,
                'inception_score_mean': is_mean,
                'inception_score_std': is_std,
                'kl_divergence_softmax': kl_soft,
                'kl_divergence_sigmoid': kl_sig
            },
            'raw_results': results
        }
        
        return result_data
        
    except Exception as e:
        print(f"❌ Error evaluating config {config_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup temporary directories
        if os.path.exists(temp_ref_dir):
            shutil.rmtree(temp_ref_dir)
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)

def main():
    print("🎵 MULTIDIFFUSION CONFIG EVALUATION")
    print("🔍 Comparing each config against AudioSet reference")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Initialize evaluator with PASST backbone
    print("🔧 Initializing PASST evaluator...")
    evaluator = EvaluationHelper(16000, device, backbone="passt")
    print("✅ Evaluator ready!")
    
    # Reference directory (AudioSet)
    reference_dir = "AudioSet/downloaded_audio"
    if not os.path.exists(reference_dir):
        print(f"❌ Reference directory not found: {reference_dir}")
        return
    
    # Find available configs
    config_dirs = glob.glob("artifacts/val/novel/config_*")
    config_ids = [int(d.split('_')[-1]) for d in config_dirs if os.path.isdir(d)]
    config_ids.sort()
    
    print(f"📂 Found configs: {config_ids}")
    
    # Evaluate each config
    all_results = []
    
    for config_id in config_ids:
        result = evaluate_config(config_id, evaluator, reference_dir)
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY - ALL CONFIGS")
    print(f"{'='*70}")
    
    if not all_results:
        print("❌ No successful evaluations!")
        return
    
    # Table header
    print(f"{'Config':<8} {'FAD':<8} {'IS Mean':<8} {'IS Std':<8} {'KL Soft':<8} {'KL Sig':<8}")
    print("-" * 60)
    
    # Table rows
    for result in all_results:
        config_id = result['config_id']
        metrics = result['metrics']
        
        fad = f"{metrics['fad']:.3f}" if isinstance(metrics['fad'], (int, float)) else "N/A"
        is_mean = f"{metrics['inception_score_mean']:.3f}" if isinstance(metrics['inception_score_mean'], (int, float)) else "N/A"
        is_std = f"{metrics['inception_score_std']:.3f}" if isinstance(metrics['inception_score_std'], (int, float)) else "N/A"
        kl_soft = f"{metrics['kl_divergence_softmax']:.3f}" if isinstance(metrics['kl_divergence_softmax'], (int, float)) else "N/A"
        kl_sig = f"{metrics['kl_divergence_sigmoid']:.3f}" if isinstance(metrics['kl_divergence_sigmoid'], (int, float)) else "N/A"
        
        print(f"{config_id:<8} {fad:<8} {is_mean:<8} {is_std:<8} {kl_soft:<8} {kl_sig:<8}")
    
    # Save results
    with open('config_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: config_evaluation_results.json")
    print("\n🏆 Best performers:")
    print("   • Lowest FAD = Best audio quality")
    print("   • Highest IS = Best diversity/quality")
    print("   • Lowest KL = Best distribution match")

if __name__ == "__main__":
    main()
