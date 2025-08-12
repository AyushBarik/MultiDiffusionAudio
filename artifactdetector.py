import numpy as np
import librosa
import librosa.display
import os
import glob
import matplotlib.pyplot as plt

# The calculate_seam_scores function remains the same as the previous version.
def calculate_seam_scores(
    audio_path: str,
    chunk_duration_sec: float,
    sr: int = 22050,
    ups_window_sec: float = 0.2,
    n_fft: int = 2048,
    hop_length: int = 512,
    epsilon: float = 1e-10
):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        if len(audio) == 0:
            return {}
    except Exception as e:
        return {}

    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    frames_per_chunk = int(chunk_duration_sec * sr / hop_length)
    if frames_per_chunk == 0:
        return {}
    num_chunks = spectrogram.shape[1] // frames_per_chunk
    if num_chunks < 2:
        return {'avg_SDS': 0, 'avg_UPS': 1.0, 'sds_per_seam': [], 'ups_per_seam': [], 'boundary_times': []}

    boundary_frames = [i * frames_per_chunk for i in range(1, num_chunks)]
    sds_per_seam = []
    for t in boundary_frames:
        if t > 0 and t < spectrogram_db.shape[1]:
            v_end, v_start = spectrogram_db[:, t - 1], spectrogram_db[:, t]
            sds_per_seam.append(np.mean((v_end - v_start) ** 2))
    ups_per_seam = []
    window_size_frames = int(ups_window_sec * sr / hop_length)
    for t_seam in boundary_frames:
        t_chunk = t_seam - (frames_per_chunk // 2)
        if (t_seam - window_size_frames // 2 >= 0 and
            t_seam + window_size_frames // 2 < spectrogram.shape[1] and
            t_chunk - window_size_frames // 2 >= 0 and
            t_chunk + window_size_frames // 2 < spectrogram.shape[1]):
            seam_window = spectrogram[:, t_seam - window_size_frames//2 : t_seam + window_size_frames//2]
            chunk_window = spectrogram[:, t_chunk - window_size_frames//2 : t_chunk + window_size_frames//2]
            e_seam, e_chunk = np.mean(seam_window), np.mean(chunk_window)
            ups_per_seam.append(e_chunk / (e_seam + epsilon))
    return {
        'avg_SDS': np.mean(sds_per_seam) if sds_per_seam else 0,
        'avg_UPS': np.mean(ups_per_seam) if ups_per_seam else 1.0,
        'sds_per_seam': sds_per_seam,
        'ups_per_seam': ups_per_seam,
        'boundary_times': [t * hop_length / sr for t in boundary_frames]
    }

def analyze_and_plot_seams(
    audio_path: str,
    chunk_duration_sec: float,
    sds_threshold: float = 15.0,
    ups_threshold: float = 2.0
):
    """
    Performs a deep-dive analysis on a single audio file, printing per-seam
    scores and saving a plot of the spectrogram with artifacts highlighted.
    """
    print(f"\n--- Analyzing file: {os.path.basename(audio_path)} ---")
    scores = calculate_seam_scores(audio_path, chunk_duration_sec)
    if not scores:
        print("Could not process file.")
        return

    print(f"Average Scores: SDS={scores['avg_SDS']:.4f}, UPS={scores['avg_UPS']:.4f}")
    print("\nPer-Seam Analysis:")
    for i, t in enumerate(scores['boundary_times']):
        sds = scores['sds_per_seam'][i]
        ups = scores['ups_per_seam'][i]
        print(f"  - Seam at {t:.2f}s: SDS={sds:.2f}, UPS={ups:.2f}")

    sr, hop_length = 22050, 512
    audio, _ = librosa.load(audio_path, sr=sr)
    spectrogram_db = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length), ref=np.max)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(ax.get_children()[0], ax=ax, format='%+2.0f dB')
    ax.set_title(f'Spectrogram and Detected Artifacts for {os.path.basename(audio_path)}')

    for i, t in enumerate(scores['boundary_times']):
        if scores['sds_per_seam'][i] > sds_threshold:
            ax.axvline(x=t, color='blue', linestyle='--', label=f'High SDS ({scores["sds_per_seam"][i]:.1f})')
        if scores['ups_per_seam'][i] > ups_threshold:
            ax.axvline(x=t, color='red', linestyle=':', label=f'High UPS ({scores["ups_per_seam"][i]:.1f})')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())
    
    output_filename = os.path.splitext(audio_path)[0] + '_analysis.png'
    plt.savefig(output_filename)
    plt.close(fig) # Close the figure to free memory
    print(f"\nAnalysis plot saved to: {output_filename}")


if __name__ == '__main__':
    target_directory = 'benchaudios' 
    chunk_duration = 10.0
    
    good_file = os.path.join(target_directory, 'multi_0.80_256.wav')
    bad_file_high_ups = os.path.join(target_directory, 'multi_0.70_256.wav')
    
    if os.path.exists(good_file):
        analyze_and_plot_seams(good_file, chunk_duration_sec=chunk_duration)
    else:
        print(f"\nWarning: Test file not found at {good_file}. Skipping analysis.")

    if os.path.exists(bad_file_high_ups):
        analyze_and_plot_seams(bad_file_high_ups, chunk_duration_sec=chunk_duration)
    else:
        print(f"\nWarning: Test file not found at {bad_file_high_ups}. Skipping analysis.")