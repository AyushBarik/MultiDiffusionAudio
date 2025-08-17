#!/usr/bin/env python3
"""
Evaluation script using the official audioldm_eval toolbox.
Calculates FD, IS, KL, and FAD metrics on chunked audio files.
"""

import os
import torch
import glob
from audioldm_eval import EvaluationHelper

def main():
    print("🎵 AudioLDM Evaluation with Official Toolbox (PASST Backbone)")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Initialize evaluator with PASST backbone
    print("🔧 Initializing EvaluationHelper with PASST backbone...")
    evaluator = EvaluationHelper(32000, device, backbone="passt", recalculate=True)
    print("✅ EvaluationHelper (PASST) initialized successfully!")
    
    # Define paths
    ref_dir = "chunked_ref"
    gen_dir = "artifacts/val/novel/config_8"
    
    print(f"\n📁 Reference audio directory: {ref_dir}")
    print(f"📁 Generated audio directory: {gen_dir}")
    
    # Check if directories exist
    if not os.path.exists(ref_dir):
        print(f"❌ Error: {ref_dir} directory not found!")
        return
    
    if not os.path.exists(gen_dir):
        print(f"❌ Error: {gen_dir} directory not found!")
        return
    
    # Count audio files
    ref_files = glob.glob(os.path.join(ref_dir, "*.wav"))
    gen_files = glob.glob(os.path.join(gen_dir, "*.wav"))
    
    print(f"📊 Found {len(ref_files)} reference files")
    print(f"📊 Found {len(gen_files)} generated files")
    
    if len(ref_files) == 0 or len(gen_files) == 0:
        print("❌ Error: No audio files found in one or both directories!")
        return
    
    print("\n🚀 Starting evaluation...")
    print("-" * 30)
    
    try:
        # Calculate all metrics using the main method
        print("📈 Calculating all metrics (FD, IS, KL, FAD)...")
        print("This may take several minutes...")
        
        # Use the main calculate_metrics method
        results = evaluator.calculate_metrics(
            generate_files_path=gen_dir,
            groundtruth_path=ref_dir,
            same_name=True,
            limit_num=None,
            calculate_psnr_ssim=False,
            calculate_lsd=False,  # LSD is disabled due to dependency issues
            recalculate=False
        )
        
        print("✅ Metrics calculation completed!")
        
        # Extract individual metrics from results
        fad_score = results.get('frechet_audio_distance', 'N/A')
        is_score = results.get('inception_score_mean', 'N/A')
        is_std = results.get('inception_score_std', 'N/A')
        kl_score = results.get('kl_divergence', 'N/A')
        fd_score = results.get('frechet_distance', 'N/A')
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 FINAL RESULTS SUMMARY")
        print("=" * 50)
        print(f"🎯 Fréchet Audio Distance (FAD): {fad_score}")
        print(f"🎯 Inception Score (IS):         {is_score} ± {is_std}")
        print(f"🎯 KL Divergence:                {kl_score}")
        print(f"🎯 Fréchet Distance (FD):        {fd_score}")
        print("=" * 50)
        
        # Print all available results
        print(f"\n📊 ALL RESULTS:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
        # Interpretation
        print("\n📊 METRIC INTERPRETATION:")
        print("• Lower FD = Better quality (closer to reference)")
        print("• Higher IS = Better diversity and quality")
        print("• Lower KL = Better distribution match")
        print("• Lower FAD = Better audio quality and realism")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()