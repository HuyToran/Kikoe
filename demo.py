#!/usr/bin/env python3
"""
Kikoe - Quick Demo
==================
Test the three pre-trained Japanese wake word models without any setup.

Models bundled in models/:
  1. ハナケア      (hanakea)       — FP=1.24/hr, Recall=0.686
  2. おやすみなさい (oyasuminasai) — FP=0.53/hr, Recall=0.683
  3. 寝てください   (netekudasai)  — FP=1.06/hr, Recall=0.697

Requirements:
  pip install openwakeword pyaudio numpy onnxruntime

Usage:
  python demo.py           # interactive model selection
  python demo.py --model hanakea
  python demo.py --model oyasuminasai --threshold 0.05
  python demo.py --all     # run all three models simultaneously
"""

import argparse
import collections
import datetime
import os
import sys
import wave

import numpy as np

MODELS = {
    "hanakea":       ("ハナケア",       "models/hanakea.onnx"),
    "oyasuminasai":  ("おやすみなさい", "models/oyasuminasai.onnx"),
    "netekudasai":   ("寝てください",   "models/netekudasai.onnx"),
}

RATE  = 16000
CHUNK = 1280


def check_models():
    missing = [k for k, (_, path) in MODELS.items() if not os.path.exists(path)]
    if missing:
        print("Error: missing model files:")
        for m in missing:
            print(f"  {MODELS[m][1]}")
        print("\nMake sure you cloned the repo with model files intact.")
        sys.exit(1)


def run_single(model_key: str, threshold: float, save_audio: bool):
    import pyaudio
    from openwakeword.model import Model

    wake_word, model_path = MODELS[model_key]
    oww    = Model(wakeword_models=[model_path], inference_framework="onnx")
    pa     = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    save_dir = f"captured_audio/{model_key}"
    if save_audio:
        os.makedirs(save_dir, exist_ok=True)

    BUFFER_CHUNKS = int(2 * RATE / CHUNK)
    audio_buffer  = collections.deque(maxlen=BUFFER_CHUNKS)
    COOLDOWN      = int(1.5 * RATE / CHUNK)
    cooldown      = 0
    pending_file  = None
    peak_score    = 0.0

    print("=" * 60)
    print(f"Listening for: {wake_word}  ({model_key})")
    print(f"  threshold = {threshold}")
    if save_audio:
        print(f"  saving    = {save_dir}/")
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


def run_all(threshold: float, save_audio: bool):
    """Run all three models simultaneously on the same audio stream."""
    import pyaudio
    from openwakeword.model import Model

    model_paths = [path for _, path in MODELS.values()]
    oww    = Model(wakeword_models=model_paths, inference_framework="onnx")
    pa     = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)

    if save_audio:
        for key in MODELS:
            os.makedirs(f"captured_audio/{key}", exist_ok=True)

    BUFFER_CHUNKS = int(2 * RATE / CHUNK)
    audio_buffer  = collections.deque(maxlen=BUFFER_CHUNKS)
    COOLDOWN      = int(1.5 * RATE / CHUNK)
    cooldowns     = {k: 0 for k in MODELS}
    peak_scores   = {k: 0.0 for k in MODELS}
    pending_files = {k: None for k in MODELS}

    labels = {
        "hanakea":       "ハナケア      ",
        "oyasuminasai":  "おやすみなさい",
        "netekudasai":   "寝てください  ",
    }

    print("=" * 60)
    print("Listening for ALL three wake words simultaneously:")
    for key, (word, _) in MODELS.items():
        print(f"  {word}")
    print(f"  threshold = {threshold}")
    print("Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        while True:
            audio = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            audio_buffer.append(audio)

            preds  = oww.predict(audio)
            volume = int(np.abs(audio).mean())

            line_parts = [f"vol={volume:4d}"]
            for key, (_, model_path) in MODELS.items():
                model_id = os.path.splitext(os.path.basename(model_path))[0]
                score    = preds.get(model_id, 0.0)

                if save_audio:
                    if score >= threshold and cooldowns[key] == 0:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        pending_files[key] = f"captured_audio/{key}/tmp_{ts}.wav"
                        peak_scores[key]   = score
                        clip = np.concatenate(list(audio_buffer))
                        with wave.open(pending_files[key], "w") as wf:
                            wf.setnchannels(1); wf.setsampwidth(2)
                            wf.setframerate(RATE); wf.writeframes(clip.tobytes())
                        cooldowns[key] = COOLDOWN

                    if cooldowns[key] > 0:
                        peak_scores[key] = max(peak_scores[key], score)
                        cooldowns[key]  -= 1
                        if cooldowns[key] == 0 and pending_files[key]:
                            base  = os.path.basename(pending_files[key]).replace("tmp_", "")
                            final = f"captured_audio/{key}/peak{peak_scores[key]:.3f}_{base}"
                            os.rename(pending_files[key], final)
                            print(f"\n[SAVED {key}] {final}")
                            pending_files[key] = None
                            peak_scores[key]   = 0.0

                flag = "DETECTED" if score > threshold else "       "
                bar  = "█" * int(score * 20)
                line_parts.append(f"{labels[key]}={score:.3f}|{bar:<20}|{flag}")

            print("\r" + "  ".join(line_parts) + "    ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def interactive_select():
    print("=" * 50)
    print("  Kikoe - Pre-trained Wake Word Demo")
    print("=" * 50)
    print("\nAvailable models:")
    keys = list(MODELS.keys())
    for i, key in enumerate(keys, 1):
        word, path = MODELS[key]
        print(f"  {i}. {word} ({key})")
    print(f"  {len(keys)+1}. All three simultaneously")
    print()
    choice = input("Select [1-4]: ").strip()
    try:
        n = int(choice)
    except ValueError:
        print("Invalid choice."); sys.exit(1)
    if 1 <= n <= len(keys):
        return keys[n - 1], False
    elif n == len(keys) + 1:
        return "all", False
    else:
        print("Invalid choice."); sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Kikoe demo — test pre-trained Japanese wake word models",
    )
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Model to use (skips interactive prompt)")
    parser.add_argument("--all", action="store_true",
                        help="Run all three models simultaneously")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (default: 0.5)")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable saving detected audio clips")
    args = parser.parse_args()

    check_models()

    save_audio = not args.no_save

    if args.all:
        run_all(args.threshold, save_audio)
    elif args.model:
        run_single(args.model, args.threshold, save_audio)
    else:
        selected, _ = interactive_select()
        if selected == "all":
            run_all(args.threshold, save_audio)
        else:
            run_single(selected, args.threshold, save_audio)


if __name__ == "__main__":
    main()
