"""
Download required training data for openWakeWord:
  1. Room Impulse Responses (MIT RIRs)
  2. Background noise (AudioSet + Free Music Archive)
  3. Pre-computed features (ACAV100M train + validation set)
  4. openWakeWord embedding models

Run once before your first training session:
  python download_data.py
"""

import os
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm

try:
    import datasets
except ImportError:
    os.system("pip install datasets")
    import datasets


# ── Room Impulse Responses ─────────────────────────────────────────────────────

def download_rirs(output_dir="./mit_rirs"):
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        print(f"RIRs already exist in {output_dir}, skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading MIT Room Impulse Responses...")
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    for row in tqdm(ds, desc="Saving RIRs"):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    print(f"RIRs saved to {output_dir}")


# ── Background noise ───────────────────────────────────────────────────────────

def download_audioset(output_dir="./audioset_16k"):
    if Path(output_dir).exists() and len(list(Path(output_dir).glob("*.wav"))) > 10:
        print(f"AudioSet already exists in {output_dir}, skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading AudioSet background noise (streaming)...")
    ds = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True
    ).cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(ds, desc="Converting AudioSet to 16kHz"):
        name = row["video_id"] + ".wav"
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    print(f"AudioSet saved to {output_dir}")


def download_fma(output_dir="./fma", n_hours=1):
    if Path(output_dir).exists() and len(list(Path(output_dir).glob("*.wav"))) > 10:
        print(f"FMA already exists in {output_dir}, skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading Free Music Archive clips...")
    import aiohttp
    ds = datasets.load_dataset(
        "rudraml/fma", name="small", split="train",
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=7200)}},
    ).cast_column("audio", datasets.Audio(sampling_rate=16000))
    n_clips = min(n_hours * 3600 // 30, len(ds))
    for i in tqdm(range(n_clips), desc="Saving FMA clips"):
        row = ds[i]
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    print(f"FMA saved to {output_dir}")


# ── Pre-computed features ──────────────────────────────────────────────────────

def download_features():
    files = {
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": (
            "https://huggingface.co/datasets/davidscripka/openwakeword_features"
            "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        ),
        "validation_set_features.npy": (
            "https://huggingface.co/datasets/davidscripka/openwakeword_features"
            "/resolve/main/validation_set_features.npy"
        ),
    }
    for fname, url in files.items():
        if not os.path.exists(fname):
            print(f"Downloading {fname}...")
            os.system(f"wget -q --show-progress -O {fname} {url}")
        else:
            print(f"Already exists: {fname}")


# ── openWakeWord embedding models ──────────────────────────────────────────────

def download_oww_models():
    model_dir = "./openWakeWord/openwakeword/resources/models"
    os.makedirs(model_dir, exist_ok=True)
    base = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
    models = [
        "embedding_model.onnx",
        "embedding_model.tflite",
        "melspectrogram.onnx",
        "melspectrogram.tflite",
    ]
    for name in models:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name}...")
            os.system(f"wget -q --show-progress -O {path} {base}/{name}")
        else:
            print(f"Already exists: {name}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading training data for openWakeWord")
    print("=" * 60)

    download_oww_models()
    download_rirs()
    download_audioset()
    download_fma()
    download_features()

    print("\n" + "=" * 60)
    print("All training data downloaded successfully!")
    print("Next step: python wakeword.py generate --config configs/your_config.yaml")
    print("=" * 60)
