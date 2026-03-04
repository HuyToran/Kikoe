#!/usr/bin/env python3
"""
Kikoe - Japanese Wake Word Training Tool
=========================================
Unified CLI for dataset generation, model training, and inference.

Usage:
  python wakeword.py generate              # Generate TTS dataset
  python wakeword.py train                 # Train the model
  python wakeword.py train --steps 30000  # Override steps
  python wakeword.py infer                 # Real-time inference from mic
  python wakeword.py record                # Record real voice samples

Examples:
  python wakeword.py generate --config configs/hanakea_config.yaml
  python wakeword.py train    --config configs/hanakea_config.yaml
  python wakeword.py train    --config configs/hanakea_config.yaml --steps 40000 --seed 267
  python wakeword.py infer    --config configs/hanakea_config.yaml --threshold 0.05
  python wakeword.py record   --config configs/hanakea_config.yaml
"""

import argparse
import os
import sys
import shutil
import datetime
import subprocess
import tempfile
import yaml


DEFAULT_CONFIG = "configs/hanakea_config.yaml"


# ── generate ───────────────────────────────────────────────────────────────────

def cmd_generate(config):
    import uuid
    import random
    import logging
    import numpy as np
    import scipy.io.wavfile
    from pathlib import Path
    from tqdm import tqdm
    from melo.api import TTS

    wake_word   = config["wake_word"]
    model_name  = config["model_name"]
    n_train     = config.get("n_samples_train", 5000)
    n_val       = config.get("n_samples_val", 1000)
    adversarial = config.get("adversarial_phrases", [])
    speeds      = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    base        = os.path.join(config.get("output_dir", "./my_custom_model"), model_name)

    dirs = {
        "pos_train": os.path.join(base, "positive_train"),
        "pos_test":  os.path.join(base, "positive_test"),
        "neg_train": os.path.join(base, "negative_train"),
        "neg_test":  os.path.join(base, "negative_test"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def gen_clips(model, spk_id, text, out_dir, n, desc):
        existing = len(list(Path(out_dir).glob("*.wav")))
        if existing >= int(n * 0.95):
            print(f"  Skipping {desc}: {existing}/{n} clips already exist")
            return
        for _ in tqdm(range(n - existing), desc=desc):
            speed  = random.choice(speeds)
            chosen = random.choice(text) if isinstance(text, list) else text
            path   = os.path.join(out_dir, f"{uuid.uuid4().hex}.wav")
            try:
                devnull = open(os.devnull, "w")
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = devnull, devnull
                logging.disable(logging.CRITICAL)
                try:
                    model.tts_to_file(chosen, spk_id, path, speed=speed)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                    devnull.close()
                    logging.disable(logging.NOTSET)
                sr, data = scipy.io.wavfile.read(path)
                if sr != 16000:
                    from scipy.signal import resample
                    data = resample(data, int(len(data) * 16000 / sr)).astype(np.int16)
                    scipy.io.wavfile.write(path, 16000, data)
            except Exception as e:
                print(f"  Error: {e}")

    print("=" * 60)
    print(f"Generating dataset for: {wake_word}")
    print(f"  train samples : {n_train}")
    print(f"  val samples   : {n_val}")
    print(f"  adversarials  : {len(adversarial)} phrases")
    print(f"  output        : {base}")
    print("=" * 60)
    print("\nLoading MeloTTS Japanese model...")
    model  = TTS(language="JP", device="auto")
    spk_id = model.hps.data.spk2id["JP"]

    gen_clips(model, spk_id, wake_word,    dirs["pos_train"], n_train, "Positive train")
    gen_clips(model, spk_id, wake_word,    dirs["pos_test"],  n_val,   "Positive val  ")
    if adversarial:
        gen_clips(model, spk_id, adversarial, dirs["neg_train"], n_train, "Negative train")
        gen_clips(model, spk_id, adversarial, dirs["neg_test"],  n_val,   "Negative val  ")

    print("\n" + "=" * 60)
    print(f"Done! Dataset ready at: {base}")
    print("Next: python wakeword.py train --config <your_config.yaml>")
    print("=" * 60)


# ── train ──────────────────────────────────────────────────────────────────────

def cmd_train(config):
    model_name   = config["model_name"]
    output_dir   = config.get("output_dir", "./my_custom_model")
    backup_dir   = config.get("backup_dir", "./model_backups")
    python_path  = config.get("python_path", sys.executable)
    train_script = config.get("train_script", "./openWakeWord/openwakeword/train.py")

    bg_paths = config.get("background_audio_dirs", ["./audioset_16k", "./fma"])
    train_cfg = {
        "model_name":          model_name,
        "target_phrase":       [config["wake_word"]],
        "custom_negative_phrases": [],
        "n_samples":           config.get("n_samples_train", 5000),
        "n_samples_val":       config.get("n_samples_val", 1000),
        "tts_batch_size":      50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": "./piper-sample-generator",
        "output_dir":          output_dir,
        "rir_paths":           config.get("rir_dirs", ["./mit_rirs"]),
        "background_paths":    bg_paths,
        "background_paths_duplication_rate": [1] * len(bg_paths),
        "false_positive_validation_data_path": config.get(
            "validation_features", "./validation_set_features.npy"),
        "augmentation_rounds": config.get("augmentation_rounds", 3),
        "feature_data_files": {
            "ACAV100M_sample": config.get(
                "acav_features", "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
        },
        "batch_n_per_class": {
            "ACAV100M_sample":      1024,
            "adversarial_negative": 50,
            "positive":             50,
        },
        "random_seed":         config.get("random_seed", 42),
        "model_type":          "dnn",
        "layer_size":          config.get("layer_size", 32),
        "steps":               config.get("steps", 20000),
        "max_negative_weight": config.get("max_negative_weight", 1500),
        "target_false_positives_per_hour": config.get(
            "target_false_positives_per_hour", 0.2),
    }

    print("=" * 60)
    print("Training configuration:")
    print(f"  wake_word   = {config['wake_word']}")
    print(f"  steps       = {train_cfg['steps']}")
    print(f"  layer_size  = {train_cfg['layer_size']}")
    print(f"  aug_rounds  = {train_cfg['augmentation_rounds']}")
    print(f"  max_neg_w   = {train_cfg['max_negative_weight']}")
    print(f"  random_seed = {train_cfg['random_seed']}")
    print(f"  output_dir  = {output_dir}")
    print("=" * 60)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(train_cfg, tmp)
    tmp.close()

    try:
        subprocess.run(
            [python_path, train_script,
             "--training_config", tmp.name,
             "--augment_clips", "--train_model"],
            check=True,
        )
    finally:
        os.unlink(tmp.name)

    # Backup model after training
    onnx = os.path.join(output_dir, f"{model_name}.onnx")
    if os.path.exists(onnx):
        os.makedirs(backup_dir, exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        aug   = train_cfg["augmentation_rounds"]
        steps = train_cfg["steps"]
        seed  = train_cfg["random_seed"]
        dst   = os.path.join(
            backup_dir,
            f"{model_name}_{ts}_aug{aug}_steps{steps}_seed{seed}.onnx",
        )
        shutil.copy2(onnx, dst)
        print("=" * 60)
        print(f"Model backup saved: {dst}")
        print("=" * 60)


# ── infer ──────────────────────────────────────────────────────────────────────

def cmd_infer(config):
    import collections
    import wave
    import numpy as np
    import pyaudio
    from openwakeword.model import Model

    model_name = config["model_name"]
    output_dir = config.get("output_dir", "./my_custom_model")
    threshold  = config.get("threshold", 0.01)
    save_audio = config.get("save_audio", True)
    save_dir   = config.get("save_dir", "captured_audio")
    RATE, CHUNK = 16000, 1280

    # Look for model in output_dir first, then models/ folder
    model_path = os.path.join(output_dir, f"{model_name}.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", f"{model_name}.onnx")
    if not os.path.exists(model_path):
        print(f"Error: model not found at {output_dir}/{model_name}.onnx or models/{model_name}.onnx")
        print("Run 'python wakeword.py train' first, or use demo.py for pre-trained models.")
        sys.exit(1)

    oww    = Model(wakeword_models=[model_path], inference_framework="onnx")
    pa     = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    if save_audio:
        os.makedirs(save_dir, exist_ok=True)

    BUFFER_CHUNKS = int(2 * RATE / CHUNK)
    audio_buffer  = collections.deque(maxlen=BUFFER_CHUNKS)
    COOLDOWN      = int(1.5 * RATE / CHUNK)
    cooldown      = 0
    pending_file  = None
    peak_score    = 0.0

    print("=" * 60)
    print(f"Listening for: {config['wake_word']}")
    print(f"  model     = {model_path}")
    print(f"  threshold = {threshold}")
    if save_audio:
        print(f"  saving    = {save_dir}/ (peak score in filename)")
    print("Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            audio = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            audio_buffer.append(audio)

            pred   = oww.predict(audio)
            score  = list(pred.values())[0]
            volume = int(np.abs(audio).mean())

            if save_audio:
                if score >= threshold and cooldown == 0:
                    ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    pending_file = os.path.join(save_dir, f"tmp_{ts}.wav")
                    peak_score   = score
                    clip         = np.concatenate(list(audio_buffer))
                    with wave.open(pending_file, "w") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2)
                        wf.setframerate(RATE); wf.writeframes(clip.tobytes())
                    cooldown = COOLDOWN

                if cooldown > 0:
                    peak_score = max(peak_score, score)
                    cooldown  -= 1
                    if cooldown == 0 and pending_file:
                        base  = os.path.basename(pending_file).replace("tmp_", "")
                        final = os.path.join(save_dir, f"peak{peak_score:.3f}_{base}")
                        os.rename(pending_file, final)
                        print(f"\n[SAVED] {final}  (peak={peak_score:.3f})")
                        pending_file = None
                        peak_score   = 0.0

            bar    = "█" * int(score * 40)
            status = ">>> DETECTED! <<<" if score > threshold else ""
            print(f"\rvol={volume:4d} | score={score:.4f} |{bar:<40}| {status}    ",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── record ─────────────────────────────────────────────────────────────────────

def cmd_record(config):
    import time
    import wave
    import numpy as np
    import pyaudio

    out_dir        = config.get("record_dir", "./my_voice_samples")
    silence_thresh = config.get("silence_threshold", 200)
    record_secs    = config.get("record_seconds", 2.5)
    pre_secs       = config.get("pre_record_seconds", 0.3)
    RATE, CHUNK    = 16000, 1280

    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    count    = existing

    pa     = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    PRE_CHUNKS = int(pre_secs * RATE / CHUNK)
    REC_CHUNKS = int(record_secs * RATE / CHUNK)
    pre_buf, rec_buf = [], []
    recording, rec_count = False, 0

    print("=" * 60)
    print(f"Recording voice samples for: {config['wake_word']}")
    print(f"  output    = {out_dir}/  (existing: {existing})")
    print(f"  threshold = vol > {silence_thresh}")
    print(f"  duration  = {record_secs}s per sample")
    print("Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            audio  = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            volume = int(np.abs(audio).mean())
            pre_buf.append(audio.copy())
            if len(pre_buf) > PRE_CHUNKS:
                pre_buf.pop(0)

            if not recording:
                print(f"\rvol={volume:4d} | Waiting for voice...", end="", flush=True)
                if volume > silence_thresh:
                    recording  = True
                    rec_buf    = list(pre_buf)
                    rec_count  = 0
                    print("\n[Recording...]")
            else:
                rec_buf.append(audio.copy())
                rec_count += 1
                print(f"\r[Recording] {rec_count}/{REC_CHUNKS} chunks", end="", flush=True)
                if rec_count >= REC_CHUNKS:
                    count += 1
                    fname = os.path.join(out_dir, f"sample_{count:04d}.wav")
                    data  = np.concatenate(rec_buf).astype(np.int16)
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2)
                        wf.setframerate(RATE); wf.writeframes(data.tobytes())
                    print(f"\n>>> Saved: {fname}\n")
                    recording = False
                    rec_buf   = []
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nDone. Saved {count - existing} new samples to '{out_dir}/'")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kikoe - Japanese Wake Word Training Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  generate   Generate TTS dataset using MeloTTS (run once per wake word)
  train      Train the wake word model with openWakeWord
  infer      Real-time inference from microphone
  record     Record real voice samples for data collection

Examples:
  python wakeword.py generate --config configs/hanakea_config.yaml
  python wakeword.py train    --config configs/hanakea_config.yaml
  python wakeword.py train    --config configs/hanakea_config.yaml --steps 40000 --seed 267
  python wakeword.py infer    --config configs/hanakea_config.yaml --threshold 0.05
  python wakeword.py record   --config configs/hanakea_config.yaml
        """,
    )
    parser.add_argument("command", choices=["generate", "train", "infer", "record"])
    parser.add_argument("--config",    default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--steps",     type=int,   help="Override training steps")
    parser.add_argument("--aug",       type=int,   help="Override augmentation rounds")
    parser.add_argument("--seed",      type=int,   help="Override random seed")
    parser.add_argument("--threshold", type=float, help="Override inference threshold")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        print("Available configs:")
        for f in sorted(os.listdir("configs") if os.path.isdir("configs") else []):
            if f.endswith(".yaml"):
                print(f"  configs/{f}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.steps     is not None: config["steps"]               = args.steps
    if args.aug       is not None: config["augmentation_rounds"] = args.aug
    if args.seed      is not None: config["random_seed"]         = args.seed
    if args.threshold is not None: config["threshold"]           = args.threshold

    if   args.command == "generate": cmd_generate(config)
    elif args.command == "train":    cmd_train(config)
    elif args.command == "infer":    cmd_infer(config)
    elif args.command == "record":   cmd_record(config)


if __name__ == "__main__":
    main()
