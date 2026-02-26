import librosa.feature
import numpy as np
import pandas as pd
import os

def extract_features(file_path, label="unknown", sr=22050, segment_duration=3):
    """
    Extract features from a song, splitting it into 3-second segments.

    Returns a DataFrame where each row corresponds to one 3-second segment.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    total_samples = len(y)
    segment_samples = segment_duration * sr

    all_data = []

    # Split into 3-second segments
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        y_segment = y[start:end]

        min_samples = segment_samples // 3
        if len(y_segment) < min_samples:
            continue

        data = {}
        # Metadata
        data["filename"] = os.path.basename(file_path)
        data["length"] = len(y_segment) / sr

        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
        data["chroma_stft_mean"] = np.mean(chroma)
        data["chroma_stft_var"] = np.var(chroma)

        # RMS
        rms = librosa.feature.rms(y=y_segment)
        data["rms_mean"] = np.mean(rms)
        data["rms_var"] = np.var(rms)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)

        data["spectral_centroid_mean"] = np.mean(centroid)
        data["spectral_centroid_var"] = np.var(centroid)
        data["spectral_bandwidth_mean"] = np.mean(bandwidth)
        data["spectral_bandwidth_var"] = np.var(bandwidth)
        data["rolloff_mean"] = np.mean(rolloff)
        data["rolloff_var"] = np.var(rolloff)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y_segment)
        data["zero_crossing_rate_mean"] = np.mean(zcr)
        data["zero_crossing_rate_var"] = np.var(zcr)

        # Harmonic & Percussive
        y_harm, y_perc = librosa.effects.hpss(y_segment)
        data["harmony_mean"] = np.mean(y_harm)
        data["harmony_var"] = np.var(y_harm)
        data["perceptr_mean"] = np.mean(y_perc)  # dataset typo preserved
        data["perceptr_var"] = np.var(y_perc)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr)
        data["tempo"] = tempo

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
        for i in range(20):
            data[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
            data[f"mfcc{i+1}_var"] = np.var(mfccs[i])

        # Label
        data["label"] = label

        all_data.append(data)

    # Define exact column order
    columns = [
        'filename', 'length', 'chroma_stft_mean', 'chroma_stft_var',
        'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var',
        'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var',
        'perceptr_mean', 'perceptr_var',
        'tempo'
    ]
    for i in range(20):
        columns.extend([f"mfcc{i+1}_mean", f"mfcc{i+1}_var"])
    columns.append("label")

    # Return as DataFrame
    return pd.DataFrame(all_data, columns=columns)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    genre = "jazz"
    number = "00000"

    file_path = f'data/genres_original/{genre}/{genre}.{number}.wav'
    custom_features = extract_features(file_path, label=genre)
    features_3_sec_path = '../data/features_3_sec.csv'
    features_3_sec = pd.read_csv(features_3_sec_path)
    predefined_features = features_3_sec[features_3_sec['filename'].str.contains('{}.{}'.format(genre, number))]
    df1 = custom_features.iloc[0:9]  # First 3 rows of your extracted features
    df2 = predefined_features.iloc[0:9]  # First 3 rows of the GTZAN dataset

    # 2. Reset indices to 0, 1, 2... so they match each other
    df1_reset = df1.reset_index(drop=True)
    df2_reset = df2.reset_index(drop=True)

    # 3. Concat and Sort
    # Because they now share indices (0, 0, 1, 1...), sorting will interleave them
    interleaved = pd.concat([df1_reset, df2_reset]).sort_index(kind='merge')

    # 4. Optional: Final reset if you want a clean 0 to 17 index
    interleaved.reset_index(drop=True)

    print(interleaved)