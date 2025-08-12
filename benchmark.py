import os
import csv
from pathlib import Path
from tqdm import tqdm
from audiobox_aesthetics.infer import initialize_predictor

# --- Configuration ---
# 1. Set the path to your directory with audio files.
AUDIO_DIR = 'output/generation'

# 2. Set the name for the output CSV file.
OUTPUT_CSV = 'aesthetics_scores.csv'

# 3. Set the batch size (how many files to process at once).
BATCH_SIZE = 32
# --- End of Configuration ---

def score_audio_directory():
    """
    Scans a directory for audio files, scores them using Audiobox Aesthetics,
    and saves the results to a CSV file.
    """
    # Check if the directory exists
    if not os.path.isdir(AUDIO_DIR):
        print(f"Error: Directory not found at '{AUDIO_DIR}'")
        return

    # Find all supported audio files
    print(f"Scanning for audio files in '{AUDIO_DIR}'...")
    supported_extensions = ('.wav', '.flac', '.mp3')
    audio_paths = sorted([p for p in Path(AUDIO_DIR).rglob('*') if p.suffix.lower() in supported_extensions])

    if not audio_paths:
        print("No audio files found in the specified directory.")
        return

    print(f"Found {len(audio_paths)} audio files to process.")

    # Initialize the predictor model (this will download it on the first run)
    print("Initializing the Audiobox Aesthetics predictor...")
    try:
        predictor = initialize_predictor()
        print("✅ Predictor initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return

    # Prepare input for the model
    # The model expects a list of dictionaries: [{"path": "/path/to/file.wav"}]
    model_input = [{"path": str(p)} for p in audio_paths]

    # Process files in batches and write to CSV
    print(f"Scoring files and saving results to '{OUTPUT_CSV}'...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'PQ', 'PC', 'CE', 'CU']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm for a progress bar over the batches
        for i in tqdm(range(0, len(model_input), BATCH_SIZE), desc="Processing Batches"):
            batch_input = model_input[i:i + BATCH_SIZE]
            
            try:
                # Get predictions for the current batch
                predictions = predictor.forward(batch_input)

                # Write results for the batch to the CSV
                for file_info, scores in zip(batch_input, predictions):
                    row = {
                        'file_path': file_info['path'],
                        'PQ': f"{scores.get('PQ', 0.0):.4f}",
                        'PC': f"{scores.get('PC', 0.0):.4f}",
                        'CE': f"{scores.get('CE', 0.0):.4f}",
                        'CU': f"{scores.get('CU', 0.0):.4f}"
                    }
                    writer.writerow(row)
            except Exception as e:
                print(f"\nError processing batch starting with {batch_input[0]['path']}: {e}")

    print(f"\n✅ Done! All scores have been saved to '{OUTPUT_CSV}'.")

if __name__ == '__main__':
    score_audio_directory()