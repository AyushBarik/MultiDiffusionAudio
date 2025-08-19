import os
import csv
from pathlib import Path
import torch
from tqdm import tqdm
from msclap import CLAP

# Config
AUDIO_DIR = 'artifacts/val/novel/config_8'
PROMPT = "Musicians are drumming. Sounds from a reed instrument are also heard. Singing and voices are in the background"
OUTPUT_CSV = 'msclap_scores.csv'
BATCH_SIZE = 16
MODEL_VERSION = '2023'

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_dir_path = Path(AUDIO_DIR)

    audio_paths = sorted([str(p) for p in audio_dir_path.rglob('*.wav')])
    if not audio_paths:
        print(f"No .wav files found in '{AUDIO_DIR}'")
        return

    model = CLAP(version=MODEL_VERSION, use_cuda=(device == 'cuda'))
    
    # Corrected: Directly use the tensor output from the model
    text_embeddings = model.get_text_embeddings([PROMPT])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'score'])

        for i in tqdm(range(0, len(audio_paths), BATCH_SIZE), desc="Scoring files"):
            batch_paths = audio_paths[i:i + BATCH_SIZE]
            try:
                # Corrected: Directly use the tensor output from the model
                audio_embeddings = model.get_audio_embeddings(batch_paths, resample=True)

                scores = torch.nn.functional.cosine_similarity(
                    audio_embeddings,
                    text_embeddings,
                    dim=1
                )

                for path, score in zip(batch_paths, scores):
                    writer.writerow([path, f"{score.item():.4f}"])

            except Exception as e:
                print(f"\nError on batch starting with {batch_paths[0]}: {e}")

    print(f"Scores saved to '{OUTPUT_CSV}'")

if __name__ == '__main__':
    main()