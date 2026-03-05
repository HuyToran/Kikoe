# Kikoe (聞こえ)

A complete pipeline for training custom Japanese wake words on Mac (Apple Silicon), built on top of [openWakeWord](https://github.com/dscripka/openWakeWord) and [MeloTTS](https://github.com/myshell-ai/MeloTTS).

**Kikoe** (聞こえ) means *"I can hear it"* in Japanese.

---

## Pre-trained Sample Models

Three production-ready models are included in [`models/`](models/):

| Wake Word | Romaji | FP/hr | Recall | Layer | Steps | Adversarials |
|---|---|---|---|---|---|---|
| ハナケア | hanakea | 1.24 | 0.686 | 256 | 40,000 | none |
| おやすみなさい | oyasuminasai | 0.53 | 0.683 | 32 | 20,000 | 20 phrases |
| 寝てください | netekudasai | 1.06 | 0.697 | 32 | 20,000 | 19 phrases |

> Target: **FP ≤ 2.0/hr** and **Recall ≥ 0.65**. All three models pass both criteria.
>
> **Note on ハナケア:** Confirmed across multiple runs that omitting adversarial phrases produces identical FP/Recall vs using 19 adversarial phrases. The bundled model and config reflect this finding.

### Quick demo (no training needed)

```bash
# Install minimal requirements
pip install openwakeword pyaudio numpy onnxruntime

# Download base models (melspectrogram + embedding — required once after install)
python -c "import openwakeword; openwakeword.utils.download_models()"

# Interactive model selection
python demo.py

# Or specify directly
python demo.py --model hanakea
python demo.py --model oyasuminasai --threshold 0.4
python demo.py --all   # all three simultaneously
```

Detected audio is saved to `captured_audio/<model>/` with the peak score in the filename.

---

## Full Pipeline

### System Requirements

- **macOS** with Apple Silicon (M1/M2/M3) or Intel
- **Conda** (Miniconda or Anaconda)
- **~30 GB free disk space** (training data)
- **Microphone** for inference/recording

> All steps below were validated on a MacBook Pro with M1 Pro.

---

### Step 1 — Environment Setup

```bash
git clone https://github.com/your-username/kikoe.git
cd kikoe

# Create conda environment
conda env create -f environment.yml
conda activate kikoe

# Clone openWakeWord (required for training)
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord && pip install -e . && cd ..

# Install MeloTTS (required for data generation)
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -c "import melo; melo.utils.download_models()"

# Install PyAudio (Mac-specific: needs portaudio first)
brew install portaudio
pip install pyaudio
```

### Step 2 — Download Training Data

This downloads ~25 GB of background audio, room impulse responses, and pre-computed features required for training. **Run once.**

```bash
python download_data.py
```

Downloads:
- `mit_rirs/` — MIT Room Impulse Responses (for reverb augmentation)
- `audioset_16k/` — Google AudioSet background noise
- `fma/` — Free Music Archive clips
- `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` — 2000h pre-computed negative features
- `validation_set_features.npy` — validation set features

### Step 3 — Generate Dataset

Generate TTS clips for your wake word using MeloTTS (Japanese speaker):

```bash
# Using one of the bundled configs
python wakeword.py generate --config configs/hanakea_config.yaml

# Or your own config (copy and edit one of the bundled ones)
python wakeword.py generate --config configs/my_wakeword_config.yaml
```

This creates:
```
my_custom_model/<model_name>/
  positive_train/   # 5000 TTS clips of the wake word
  positive_test/    # 1000 validation clips
  negative_train/   # 5000 adversarial-negative clips
  negative_test/    # 1000 adversarial-negative validation clips
```

**Time:** ~15-30 minutes for 5000+1000 clips on M1 Pro.

### Step 4 — Train the Model

```bash
python wakeword.py train --config configs/hanakea_config.yaml
```

Override any parameter at the command line:

```bash
python wakeword.py train --config configs/hanakea_config.yaml \
  --steps 40000 \
  --seed 267
```

After training, the model is saved to `my_custom_model/<model_name>.onnx` and automatically backed up to `model_backups/` with a timestamped filename.

**Time:** ~20-40 minutes per run on M1 Pro (varies by steps and layer size).

### Step 5 — Evaluate

The training script prints FP/hr and Recall at the end. Target values:

| Metric | Target |
|---|---|
| False Positives/hr | ≤ 2.0 |
| Recall | ≥ 0.65 |

If the model does not pass, see [docs/tuning_guide.md](docs/tuning_guide.md).

### Step 6 — Real-time Inference

```bash
# Uses the trained model in my_custom_model/
python wakeword.py infer --config configs/hanakea_config.yaml

# Lower threshold to be more sensitive (more detections, more false positives)
python wakeword.py infer --config configs/hanakea_config.yaml --threshold 0.3
```

### Optional — Record Your Own Voice Samples

Capture real microphone recordings for data analysis or future fine-tuning:

```bash
python wakeword.py record --config configs/hanakea_config.yaml
```

Recordings are saved to `my_voice_samples/<model>/sample_NNNN.wav`.

---

## Training Your Own Wake Word

1. Copy an existing config:
   ```bash
   cp configs/oyasuminasai_config.yaml configs/my_wakeword_config.yaml
   ```

2. Edit the config — at minimum change:
   ```yaml
   wake_word: "あなたのウェイクワード"
   model_name: "my_wakeword"
   adversarial_phrases:
     - "similar phrase 1"
     - "similar phrase 2"
   ```

3. Run the pipeline:
   ```bash
   python wakeword.py generate --config configs/my_wakeword_config.yaml
   python wakeword.py train    --config configs/my_wakeword_config.yaml
   python wakeword.py infer    --config configs/my_wakeword_config.yaml
   ```

See [docs/tuning_guide.md](docs/tuning_guide.md) for hyperparameter tuning tips derived from training all three bundled models.

---

## Repository Structure

```
kikoe/
├── models/                      # Pre-trained sample models (ONNX)
│   ├── hanakea.onnx             # ハナケア
│   ├── oyasuminasai.onnx        # おやすみなさい
│   └── netekudasai.onnx         # 寝てください
│
├── configs/                     # Training configs for each model
│   ├── hanakea_config.yaml
│   ├── oyasuminasai_config.yaml
│   └── netekudasai_config.yaml
│
├── demo.py                      # Quick test with pre-trained models
├── wakeword.py                  # Unified CLI: generate / train / infer / record
├── download_data.py             # One-time training data download
├── environment.yml              # Conda environment
│
└── docs/
    └── tuning_guide.md          # Hyperparameter tuning notes
```

### Key paths after full setup

```
kikoe/
├── openWakeWord/                # cloned — contains train.py
├── mit_rirs/                    # downloaded
├── audioset_16k/                # downloaded
├── fma/                         # downloaded
├── openwakeword_features_ACAV100M_2000_hrs_16bit.npy
├── validation_set_features.npy
└── my_custom_model/             # created during training
    └── <model_name>.onnx
```

---

## Mac-specific Notes

**portaudio / PyAudio**
```bash
brew install portaudio
pip install pyaudio
```

**Apple Silicon (M1/M2/M3)**
- MeloTTS uses `device="auto"` which selects MPS (Metal) automatically.
- `onnxruntime` runs on CPU by default; this is fine for inference.
- Conda is recommended over venv for managing ARM vs x86 dependencies.

**Microphone permissions**

On macOS you may need to grant Terminal (or your IDE) microphone access:
System Settings → Privacy & Security → Microphone → enable for Terminal.

---

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka — wake word detection framework
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) by MyShell — high-quality Japanese TTS for data generation
- Training data: MIT RIRs, AudioSet (Google), FMA, ACAV100M features (davidscripka/openwakeword_features)
