import librosa
import numpy as np
from scipy.stats import skew, entropy
import os
import pandas as pd
import soundfile as sf
import uuid
from tqdm import tqdm

def _already_flattened(root_path: str) -> bool:
    """
    Check if the audio samples have already been flattened.

    Args:
        root_path (str): Root directory containing 'canonical' audio files and 'labels.csv'.

    Returns:
        bool: True if data is already flattened, False otherwise.
    """
    canonical_path = os.path.join(root_path, "canonical")
    os.makedirs(canonical_path, exist_ok=True)
    len_files = len([f for f in os.listdir(canonical_path) if os.path.isfile(os.path.join(canonical_path, f))])
    
    labels_path = os.path.join(root_path, "labels.csv")
    len_labels = len(pd.read_csv(labels_path)) if os.path.exists(labels_path) else 0
    
    return len_files > 0 and len_labels == len_files


def flatten_samples(root_path: str):
    """
    Flatten all audio samples under the 'raw' subdirectory of the given root path.
    Saves flattened audio files to the 'canonical' subdirectory.
    Creates labels.csv mapping unique IDs to original file paths.

    Args:
        root_path (str): Root directory containing 'raw' audio files.

    Returns:
        None
    """

    if _already_flattened(root_path):
        print("Data already flattened. Skipping...")
        return

    labels = []
    for root, _, files in os.walk(os.path.join(root_path, "raw")):
        for file in files:
            if file.endswith(".wav"):

                # Construct Paths
                wav_path = os.path.join(
                    root, file
                )  
                rel_path = os.path.relpath(
                    wav_path, os.path.join(root_path, "raw")
                )  
                path_parts = rel_path.split(os.sep)

                # Extract category structure from path
                source = path_parts[0] if len(path_parts) > 1 else "unknown"
                category = path_parts[1] if len(path_parts) > 2 else "unknown"
                sub_category = path_parts[2] if len(path_parts) > 3 else "unknown"

                # Generate unique ID
                sample_id = str(uuid.uuid4())

                # Load and flatten audio
                y, sr = librosa.load(wav_path, sr=22050, mono=True)

                # Create label entry
                labels.append(
                    {
                        "id": sample_id,
                        "category": category,
                        "sub_category": sub_category,
                        "temporal": (
                            "loop" if sub_category.lower() == "loops" else "one-shot"
                        ),
                        "source": source,
                    }
                )

                # save flattened audio to /data/canonical
                canonical_path = os.path.join(root_path, "canonical")
                os.makedirs(canonical_path, exist_ok=True)
                canonical_wav_path = os.path.join(canonical_path, f"{sample_id}.wav")
                sf.write(canonical_wav_path, y, sr)

    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(os.path.join(root_path, "labels.csv"), index=False)


def extract_qc_features(path, sr=22050, n_fft=1024, hop_length=512):
    y, sr = librosa.load(path, sr=sr, mono=True)

    duration = len(y) / sr

    y_trim, _ = librosa.effects.trim(y, top_db=40)
    silence_ratio = 1 - (len(y_trim) / len(y)) if len(y) > 0 else 0

    rms = librosa.feature.rms(y=y_trim, hop_length=hop_length)[0]

    centroid = librosa.feature.spectral_centroid(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y_trim, n_fft=n_fft, hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(y_trim, hop_length=hop_length)[0]

    mfcc = librosa.feature.mfcc(
        y=y_trim, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )

    features = {
        "duration": duration,
        "silence_ratio": silence_ratio,
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "peak_amp": np.max(np.abs(y_trim)) if len(y_trim) else 0.0,
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
        "bandwidth_mean": np.mean(bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "flatness_mean": np.mean(flatness),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
    }

    for i in range(13):
        features[f"mfcc_{i+1:02d}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1:02d}_std"] = np.std(mfcc[i])

    return features


def extract_full_features(
    path: str,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    trim_db: int = 40,
    n_mels: int = 64
) -> tuple:
    """Extract comprehensive audio features from a single audio file.

    Args:
        path (str): Path to the audio file.
        sr (int, optional): Target sampling rate in Hz. Defaults to 22050.
        n_fft (int, optional): FFT window size for spectral analysis. Defaults to 1024.
        hop_length (int, optional): Number of samples between successive frames. Defaults to 256.
        trim_db (int, optional): Threshold in dB below reference for trimming silence. Defaults to 50.
        n_mels (int, optional): Number of mel bands for mel spectrogram. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary of extracted audio features
            - numpy.ndarray: Trimmed audio signal
            - int: Sample rate
    """
    
    y, sr = librosa.load(path, sr=sr, mono=True)

    # =========================
    # Trimming & structure
    # =========================
    y_trim, idx = librosa.effects.trim(y, top_db=trim_db)

    duration = len(y) / sr
    trimmed_duration = len(y_trim) / sr
    silence_ratio = 1.0 - (trimmed_duration / duration if duration > 0 else 0)

    trim_start_sec = idx[0] / sr
    trim_end_sec = idx[1] / sr

    # =========================
    # RMS envelope
    # =========================
    rms = librosa.feature.rms(
        y=y_trim, hop_length=hop_length
    )[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # =========================
    # Attack time (transient strength)
    # =========================
    if len(rms) > 0 and rms_mean > 0:
        peak_idx = np.argmax(rms)
        attack_time = peak_idx * hop_length / sr
    else:
        attack_time = 0.0

    # =========================
    # Spectral features
    # =========================
    centroid = librosa.feature.spectral_centroid(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y_trim, n_fft=n_fft, hop_length=hop_length
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    # =========================
    # Spectral flux (temporal change)
    # =========================
    spectral_flux = librosa.onset.onset_strength(
        y=y_trim, sr=sr, hop_length=hop_length
    )

    # =========================
    # Zero-crossing rate
    # =========================
    zcr = librosa.feature.zero_crossing_rate(
        y_trim, hop_length=hop_length
    )[0]

    # =========================
    # Temporal centroid
    # =========================
    frame_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )

    temporal_centroid = (
        np.sum(frame_times * rms) / np.sum(rms)
        if np.sum(rms) > 0 else 0.0
    )

    # =========================
    # Directionality (slopes)
    # =========================
    def slope(x):
        if len(x) < 2:
            return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    rms_slope = slope(rms)
    centroid_slope = slope(centroid)

    # =========================
    # Early vs late energy
    # =========================
    mid = len(rms) // 2
    early_energy = np.sum(rms[:mid])
    late_energy = np.sum(rms[mid:])
    early_late_energy_ratio = (
        late_energy / early_energy if early_energy > 0 else 0.0
    )

    # =========================
    # Harmonic / percussive
    # =========================
    y_harm, y_perc = librosa.effects.hpss(y_trim)

    harm_rms = np.mean(librosa.feature.rms(y=y_harm))
    perc_rms = np.mean(librosa.feature.rms(y=y_perc))

    harmonic_percussive_ratio = (
        harm_rms / perc_rms if perc_rms > 0 else 0.0
    )

    # =========================
    # Onsets & rhythm
    # =========================
    onsets = librosa.onset.onset_detect(
        y=y_trim, sr=sr, hop_length=hop_length, units="time"
    )

    onset_count = len(onsets)
    onset_density = onset_count / trimmed_duration if trimmed_duration > 0 else 0.0
    mean_inter_onset_interval = (
        np.mean(np.diff(onsets)) if len(onsets) > 1 else 0.0
    )

    # =========================
    # Pitch (pyin)
    # =========================
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_trim,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=hop_length
    )

    f0_clean = f0[~np.isnan(f0)]
    pitch_conf = voiced_prob[~np.isnan(f0)]

    # =========================
    # Chroma
    # =========================
    chroma = librosa.feature.chroma_stft(
        y=y_trim, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    chroma_entropy = entropy(chroma_mean + 1e-8)

    # =========================
    # MFCCs
    # =========================
    mfcc = librosa.feature.mfcc(
        y=y_trim, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )

    # =========================
    # Log-mel spectrogram (NEW)
    # =========================
    mel = librosa.feature.melspectrogram(
        y=y_trim, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel)

    # =========================
    # Assemble feature row
    # =========================
    row = {
        # structure
        "duration_sec": duration,
        "trimmed_duration_sec": trimmed_duration,
        "silence_ratio": silence_ratio,
        "trim_start_sec": trim_start_sec,
        "trim_end_sec": trim_end_sec,

        # energy
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "peak_amplitude": float(np.max(np.abs(y_trim))),
        "crest_factor": (
            float(np.max(np.abs(y_trim)) / rms_mean)
            if rms_mean > 0 else 0.0
        ),
        "attack_time_sec": attack_time,

        # texture
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),

        # envelope
        "temporal_centroid": temporal_centroid,
        "rms_slope": rms_slope,
        "spectral_centroid_slope": centroid_slope,
        "early_late_energy_ratio": early_late_energy_ratio,

        # spectral shape
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_centroid_skew": float(skew(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),

        # spectral texture
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_contrast_std": float(np.std(contrast)),
        "spectral_flux_mean": float(np.mean(spectral_flux)),
        "spectral_flux_std": float(np.std(spectral_flux)),

        # harmonicity
        "harmonic_rms": float(harm_rms),
        "percussive_rms": float(perc_rms),
        "harmonic_percussive_ratio": float(harmonic_percussive_ratio),

        # rhythm
        "onset_count": int(onset_count),
        "onset_density": float(onset_density),
        "mean_inter_onset_interval": float(mean_inter_onset_interval),

        # pitch
        "f0_mean": float(np.mean(f0_clean)) if len(f0_clean) else 0.0,
        "f0_std": float(np.std(f0_clean)) if len(f0_clean) else 0.0,
        "pitch_confidence_mean": float(np.mean(pitch_conf)) if len(pitch_conf) else 0.0,
        "pitch_confidence_std": float(np.std(pitch_conf)) if len(pitch_conf) else 0.0,

        # chroma
        "chroma_entropy": float(chroma_entropy),

        # mel
        "log_mel_mean": float(np.mean(log_mel)),
        "log_mel_std": float(np.std(log_mel)),
    }

    # individual chroma bins
    chroma_labels = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
    for i, note in enumerate(chroma_labels):
        row[f"chroma_mean_{note}"] = float(chroma_mean[i])
        row[f"chroma_std_{note}"] = float(chroma_std[i])

    # MFCC stats
    for i in range(mfcc.shape[0]):
        row[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        row[f"mfcc_{i+1:02d}_std"] = float(np.std(mfcc[i]))

    return row


def extract_features(wav_path: str, mode: str = "qc") -> dict:
    """
    Extract audio features from a given .wav file.

    Args:
        wav_path (str): Path to the .wav audio file.
        mode (str, optional): Feature extraction mode ('qc' or 'full'). Defaults to 'qc'.

    Returns:
        dict: Dictionary of extracted audio features.
    """
    if mode == "qc":
        features = extract_qc_features(wav_path)
    elif mode == "full":
        features = extract_full_features(wav_path)
    else:
        raise ValueError("Invalid mode. Choose 'qc' or 'full'.")

    return features


def _already_processed(root_path: str, mode: str = "qc") -> bool:
    
    features_path = os.path.join(root_path, f"features_{mode}.csv")
    return os.path.exists(features_path)


def process_samples(root_path: str, mode: str = "qc"):
    """Extract features from all canonical audio samples in the given root path.

    Args:
        root_path (str): Root directory containing 'canonical' audio files.
        mode (str, optional): Feature extraction mode ('qc' or 'full'). Defaults to "qc".
    """
    
    if _already_processed(root_path, mode=mode):
        print(f"Features already extracted in mode '{mode}'. Skipping...")
        return
    
    rows = []
    canonical_dir = os.path.join(root_path, "canonical")
    wav_files = [file for file in os.listdir(canonical_dir) if file.endswith(".wav")]
    for file in tqdm(wav_files, desc="Processing samples"):
        sample_id = file.replace(".wav", "")
        wav_path = os.path.join(canonical_dir, file)
        row = extract_features(wav_path, mode=mode)
        row["id"] = sample_id
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root_path, f"features_{mode}.csv"), index=False)
