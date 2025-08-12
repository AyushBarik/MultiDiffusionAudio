import os
import csv
from pathlib import Path
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import laion_clap

# --- Configuration ---
# Directory containing the audio files to be scored.
AUDIO_DIR = 'output/generation'

# PAM prompts for quality assessment (positive vs. negative)
POSITIVE_PROMPT = "the sound is clear and clean"
NEGATIVE_PROMPT = "the sound is noisy and with artifacts"

# The name for the output CSV file.
OUTPUT_CSV = 'pam_scores.csv'

BATCH_SIZE = 16

# Path to the custom checkpoint (download from https://huggingface.co/lukewys/laion_clap)
CUSTOM_CKPT = 'CLAP90.14.pt'  # Replace with your downloaded file path
# --- End of Configuration ---

def score_audio_with_pam():
    """
    Scans a directory for audio, scores each file using PAM (Prompting Audio-Language Models)
    for quality assessment with LAION CLAP and the music_speech_audioset_epoch_15_esc_89.98.pt checkpoint,
    and saves the results.
    """
    audio_dir_path = Path(AUDIO_DIR)
    if not audio_dir_path.is_dir():
        print(f"Error: Directory not found at '{AUDIO_DIR}'")
        return

    print(f"Scanning for audio files in '{audio_dir_path}'...")
    supported_extensions = ('.wav', '.flac', '.mp3')
    # Get a list of string paths
    audio_paths = sorted([str(p) for p in audio_dir_path.rglob('*') if p.suffix.lower() in supported_extensions])

    if not audio_paths:
        print("No audio files found.")
        return

    print(f"Found {len(audio_paths)} audio files.")

    print("Initializing LAION CLAP model with music_speech_audioset checkpoint...")
    try:
        # Load the model with HTSAT-base (required for larger checkpoints like this one)
        clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        
        # Load the specified checkpoint
        if not os.path.exists(CUSTOM_CKPT):
            print(f"❌ Checkpoint not found at '{CUSTOM_CKPT}'. Please download it from https://huggingface.co/lukewys/laion_clap and update the path.")
            return
        clap_model.load_ckpt(ckpt=CUSTOM_CKPT)
        
        print("✅ LAION CLAP model ready with music_speech_audioset checkpoint.")
    except Exception as e:
        print(f"❌ Failed to initialize LAION CLAP model: {e}")
        print("Ensure torchvision and compatible torch versions are installed. Run: pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html")
        return
        
    # Pre-calculate the text embeddings once (as tensor)
    text_embeddings = clap_model.get_text_embedding([POSITIVE_PROMPT, NEGATIVE_PROMPT], use_tensor=True)

    print(f"Scoring files and saving results to '{OUTPUT_CSV}'...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'pam_score'])

        # Process file paths in batches for efficiency
        for i in tqdm(range(0, len(audio_paths), BATCH_SIZE), desc="Processing Batches"):
            batch_paths = audio_paths[i:i + BATCH_SIZE]
            
            try:
                # Get audio embeddings directly from file list (returns tensor)
                audio_embeddings = clap_model.get_audio_embedding_from_filelist(x=batch_paths, use_tensor=True)
                
                # Compute cosine similarities to both prompts (batch x 2)
                sims = torch.nn.functional.cosine_similarity(
                    audio_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=2
                )
                
                # Apply softmax and take probability for positive prompt
                scores = torch.softmax(sims, dim=1)[:, 0]
                
                # Write results for the batch to the CSV
                for path, score in zip(batch_paths, scores):
                    writer.writerow([path, f"{score.item():.4f}"])

            except Exception as e:
                print(f"\n❌ Error processing batch starting with {batch_paths[0]}. Skipping. Error: {e}")

    print(f"\n✅ Done! Scores saved to '{OUTPUT_CSV}'.")

if __name__ == '__main__':
    score_audio_with_pam()