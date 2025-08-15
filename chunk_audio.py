#!/usr/bin/env python3
"""
Chunk long audio files into 10-second segments for proper PANNs evaluation
"""

import os
import torchaudio
import soundfile as sf
import numpy as np

def chunk_audio(input_dir, output_dir, chunk_length_sec=10, sr=16000):
    """Chunk audio files into fixed-length segments"""
    os.makedirs(output_dir, exist_ok=True)
    
    total_chunks = 0
    
    for file in os.listdir(input_dir):
        if file.endswith('.wav'):
            print(f"Processing {file}...")
            
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
            
            print(f"  → {file_chunks} chunks created")
    
    print(f"\nTotal: {total_chunks} chunks in {output_dir}")

def main():
    print("🔧 CHUNKING AUDIO FILES FOR EVALUATION")
    print("=" * 50)
    
    # Chunk reference files (config_1)
    print("\n📁 Chunking reference files...")
    chunk_audio('artifacts/val/novel/config_1', 'chunked_ref')
    
    # Chunk generated files (config_8)  
    print("\n📁 Chunking generated files...")
    chunk_audio('artifacts/val/novel/config_8', 'chunked_gen')
    
    print("\n✅ Chunking complete!")
    print("   Reference chunks: chunked_ref/")
    print("   Generated chunks: chunked_gen/")

if __name__ == "__main__":
    main()
