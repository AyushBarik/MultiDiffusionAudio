import time
import requests
import pandas as pd
import json
from io import BytesIO
from tinytag import TinyTag

FREESOUND_API_KEY = "PASTE_YOUR_FREESOUND_API_KEY_HERE"

def get_duration_from_freesound(sound_id, api_key):
    if not sound_id or api_key == "PASTE_YOUR_FREESOUND_API_KEY_HERE":
        return None
    url = f"https://freesound.org/apiv2/sounds/{sound_id}/?token={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get('duration')
    except requests.RequestException:
        return None

def get_duration_from_audio_url(href):
    if not href:
        return None
    try:
        with requests.get(href, stream=True, timeout=10) as response:
            response.raise_for_status()
            buffer = response.raw.read(32768)
            tag = TinyTag.get(BytesIO(buffer))
            return tag.duration
    except (requests.RequestException, Exception):
        return None

def get_wavcaps_samples(min_dur=60, max_dur=180, num_samples=50, output_file='wavcaps_set.csv'):
    
    urls = {
        "BBC_Sound_Effects": "https://huggingface.co/datasets/cvssp/WavCaps/raw/main/json_files/BBC_Sound_Effects.jsonl",
        "SoundBible": "https://huggingface.co/datasets/cvssp/WavCaps/raw/main/json_files/SoundBible.jsonl",
        "FreeSound": "https://huggingface.co/datasets/cvssp/WavCaps/raw/main/json_files/FreeSound.jsonl"
    }

    if FREESOUND_API_KEY == "PASTE_YOUR_FREESOUND_API_KEY_HERE":
        print("⚠️ Skipping FreeSound: API key not set in the script.")
        del urls["FreeSound"]
    
    eligible_clips = []

    for source, url in urls.items():
        print(f"Processing {source} from URL...")
        try:
            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    
                    if len(eligible_clips) >= num_samples * 10:
                        break

                    item = json.loads(line)
                    dur = None

                    if source == "BBC_Sound_Effects":
                        dur = item.get('duration')
                    elif source == "SoundBible":
                        dur = get_duration_from_audio_url(item.get('href'))
                        time.sleep(0.2)
                    elif source == "FreeSound":
                        dur = get_duration_from_freesound(item.get('id'), FREESOUND_API_KEY)
                        time.sleep(0.2)

                    if dur and min_dur <= dur <= max_dur:
                        print(f"  Found eligible clip: {item.get('id', '')} from {source} ({dur:.1f}s)")
                        clip_info = {
                            'id': str(item.get('id', '')),
                            'source': source,
                            'caption': item.get('caption', ''),
                            'duration': dur,
                            'url': item.get('href', '')
                        }
                        eligible_clips.append(clip_info)
            if len(eligible_clips) >= num_samples * 10:
                break
        except requests.RequestException as e:
            print(f"❌ Failed to download file for {source}. It may be a network issue. Error: {e}")
            continue
            
    if not eligible_clips:
        print("\nNo eligible clips found.")
        return
        
    print(f"\nFound {len(eligible_clips)} total eligible clips. Sampling {num_samples}...")
    df = pd.DataFrame(eligible_clips)
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)

    sample_df.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'.")


if __name__ == '__main__':
    get_wavcaps_samples(num_samples=10, output_file='wavcaps_tune_set.csv')
    print("-" * 30)
    get_wavcaps_samples(num_samples=50, output_file='wavcaps_eval_set.csv')