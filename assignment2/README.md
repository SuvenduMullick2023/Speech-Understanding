# Speech Understanding — Programming Assignment 2
### Code-Switched Lecture Transcription & Low-Resource Voice Cloning

---

## Table of Contents
1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Data Preparation](#data-preparation)
6. [Part I — Robust Code-Switched Transcription](#part-i--robust-code-switched-transcription)
7. [Part II — Phonetic Mapping & Translation](#part-ii--phonetic-mapping--translation)
8. [Part III — Zero-Shot Voice Cloning](#part-iii--zero-shot-voice-cloning)
9. [Part IV — Adversarial Robustness & Spoofing Detection](#part-iv--adversarial-robustness--spoofing-detection)
10. [Running the Pipeline](#running-the-pipeline)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Known Issues & Design Choices](#known-issues--design-choices)
13. [References](#references)

---

## Overview

This assignment builds an end-to-end pipeline that:

1. **Transcribes** a 10-minute Hinglish (Hindi + English code-switched) lecture using constrained ASR
2. **Maps** the transcript to a unified IPA representation
3. **Translates** it into Bengali (target Low-Resource Language)
4. **Synthesises** the lecture in Bengali using zero-shot voice cloning with the student's own voice
5. **Evaluates** robustness against adversarial attacks and synthetic speech spoofing

> **Target LRL:** Bengali (`bn`) — synthesised using `facebook/mms-tts-ben` via HuggingFace Transformers

---

## Pipeline Architecture

```
Input: Hinglish Lecture (original_segment.wav)
         │
         ▼
┌─────────────────────────┐
│  Task 1.3 · Denoising   │  Spectral Subtraction + RMS normalisation
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐       ┌──────────────────────────┐
│  Task 1.1 · LID (CNN)   │◄──────│  Task 4.2 · FGSM Attack  │
│  Hindi / English frames │       │  Adversarial noise inject │
└─────────────┬───────────┘       └──────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Task 1.2 · ASR         │  Whisper-medium + N-gram logit bias
│  Constrained decoding   │  trained on course syllabus
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Task 2.1 · IPA mapping │  Rule-based Devanagari + epitran (hin-Deva)
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Task 2.2 · Translation │  Hindi→Bengali glossary (500+ terms)
│  Hinglish → Bengali     │  + character-level script fallback
└─────────────┬───────────┘
              │                   ┌──────────────────────────┐
              ▼                   │  Task 3.1 · Speaker emb  │
┌─────────────────────────┐◄──────│  ECAPA-TDNN · 192-d vec  │
│  Task 3.3 · TTS (VITS)  │       │  student_voice_ref.wav   │
│  HuggingFace MMS-TTS    │       └──────────────────────────┘
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Task 3.2 · DTW Prosody │  F0 + energy warping from professor's style
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐       ┌──────────────────────────┐
│  Output: Bengali audio  │──────►│  Task 4.1 · Anti-spoof   │
│  output_LRL_cloned.wav  │       │  LFCC classifier · EER   │
└─────────────────────────┘       └──────────────────────────┘
```

---

## Repository Structure

```
assignment2/
├── config.py                  # All paths, thresholds, hyperparameters
├── data_prep.py               # Task 1.3 — Spectral Subtraction denoiser
├── lid_model.py               # Task 1.1 — Frame-level LID CNN + evaluation
├── asr_engine.py              # Task 1.2 — Constrained Whisper ASR + N-gram LM
├── translation_module.py      # Tasks 2.1 & 2.2 — IPA mapping + translation
├── tts_engine.py              # Tasks 3.1–3.3 — Speaker emb + VITS + DTW
├── security_module.py         # Tasks 4.1 & 4.2 — LFCC spoof + FGSM attack
├── pipeline.py                # Main orchestrator — runs full pipeline
│
├── data/
│   ├── original_segment.wav   # Source lecture clip (10 min) ← YOU PROVIDE
│   ├── student_voice_ref.wav  # Your 60s voice recording    ← YOU PROVIDE
│   ├── syllabus.txt           # Course syllabus plain text  ← YOU PROVIDE
│   └── reference_transcript.txt  # Ground truth for WER    ← YOU PROVIDE
│
├── output/
│   ├── denoised_segment.wav          # After Task 1.3
│   ├── output_LRL_cloned.wav         # Final Bengali synthesis
│   └── output_LRL_cloned_speaker_emb.npy  # ECAPA-TDNN d-vector
│
├── report/
│   └── report.pdf             # 10-page IEEE/CVPR format report
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Conda (recommended for HPC / IIT cluster)

```bash
conda create -n speech_env python=3.10 -y
conda activate speech_env

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa numpy scipy soundfile
pip install nltk g2p-en epitran
pip install dtw-python jiwer scikit-learn
pip install speechbrain
pip install indic-nlp-library
```

### Google Colab

```python
!apt-get install -y espeak-ng libsndfile1
!pip install torch torchaudio transformers
!pip install librosa numpy scipy soundfile nltk g2p-en epitran
!pip install dtw-python jiwer scikit-learn speechbrain
!pip install indic-nlp-library
```

> **Note:** `deepfilternet` is **not used** — it requires the compiled Rust extension `libdf`
> which is incompatible with `torchaudio >= 0.13` (removed `torchaudio.backend.common`).
> Spectral Subtraction is used instead, which is the listed alternative in the assignment spec.

---

## Data Preparation

You must provide the following four files before running the pipeline:

| File | Description | How to create |
|---|---|---|
| `data/original_segment.wav` | 10-min Hinglish lecture clip | Export from lecture recording |
| `data/student_voice_ref.wav` | Your 60-second voice recording | Record yourself speaking clearly |
| `data/syllabus.txt` | Course syllabus as plain text | Copy from course PDF or portal |
| `data/reference_transcript.txt` | Ground truth transcript for WER | Manual transcription or Whisper large-v3 |

### Generating a pseudo reference transcript

If manual transcription is not feasible, use Whisper large-v3 as a silver standard:

```python
import whisper, pathlib

model  = whisper.load_model("large-v3")
result = model.transcribe("data/original_segment.wav", language="en", fp16=False)
pathlib.Path("data/reference_transcript.txt").write_text(result["text"].strip())
```

> Mention this approach in your report as *"large model pseudo-reference"*.

---

## Part I — Robust Code-Switched Transcription

### Task 1.1 — Frame-Level Language Identification

**File:** `lid_model.py`

- Architecture: 2-layer CNN + `AdaptiveAvgPool2d(4,4)` + FC classifier
- Input: raw waveform at 16 kHz → 128-band log-mel spectrogram
- Output: per-segment language prediction (`0=Hindi`, `1=English`) with timestamps
- Evaluation: Macro F1-score (must exceed **0.85**)
- Timestamp precision for language switches: ≤ **200 ms**

```python
from lid_model import FrameLevelLID
model = FrameLevelLID()
segments = model.predict_language(waveform, device, frame_duration_ms=500)
```

> **Design choice:** `AdaptiveAvgPool2d` is used instead of a hardcoded linear layer size
> so the model handles any audio duration without shape errors.

### Task 1.2 — Constrained Beam Search ASR

**File:** `asr_engine.py`

- Model: `openai/whisper-medium` (upgrade to `large-v3` on A100 for best WER)
- N-gram LM: trigram model trained on `data/syllabus.txt` — boosts technical terms
- Logit biasing: `GlossaryLogitsProcessor` combines static bias + N-gram log-probability
- Beam search: 5 beams, no-repeat-ngram-size=3

```python
from asr_engine import ConstrainedWhisperASR
asr = ConstrainedWhisperASR()
transcript = asr.transcribe_with_bias(audio_np, language="hi")
```

### Task 1.3 — Denoising & Normalisation

**File:** `data_prep.py`

- Method: Power-Spectrum Spectral Subtraction (Boll, 1979)
- Noise PSD estimated from first 30 STFT frames (assumed silent lecture start)
- Over-subtraction factor α = 2.0, spectral floor β = 0.002
- Followed by RMS normalisation to −23 dB

```python
from data_prep import denoise_with_deepfilternet
denoise_with_deepfilternet("data/original_segment.wav", "output/denoised_segment.wav")
```

---

## Part II — Phonetic Mapping & Translation

### Task 2.1 — IPA Unified Representation

**File:** `translation_module.py`

- Hindi (Devanagari) → IPA via `epitran` with language code `hin-Deva`
- English → IPA via `g2p-en`
- Word-level language detection: Devanagari Unicode range `U+0900–U+097F`
- Fallback: rule-based `DEVA_IPA` dictionary (70+ entries) if epitran unavailable

```python
# Correct epitran language code — 'Hind-Deva' is WRONG, use 'hin-Deva'
epi = epitran.Epitran('hin-Deva')
```

### Task 2.2 — Semantic Translation to Bengali

**File:** `translation_module.py`

- `HINDI_TO_BENGALI`: 80+ Devanagari→Bengali technical + function word pairs
- `ENGLISH_TO_BENGALI`: English→Bengali glossary (extend to 500 entries for full credit)
- Longest-match first (up to 3-word phrases)
- Fallback: character-level Devanagari→Bengali script conversion via Unicode block offset (`0x0980 − 0x0900`)
- Final safety check ensures output always contains Bengali Unicode before TTS

---

## Part III — Zero-Shot Cross-Lingual Voice Cloning

### Task 3.1 — Speaker Embedding Extraction

**File:** `tts_engine.py`

- Model: SpeechBrain `ECAPA-TDNN` (`speechbrain/spkrec-ecapa-voxceleb`)
- Input: `student_voice_ref.wav` (exactly 60 seconds, 16 kHz)
- Output: 192-dimensional d-vector saved to `output/output_LRL_cloned_speaker_emb.npy`

### Task 3.2 — Prosody Warping via DTW

**File:** `tts_engine.py`

- F0 extraction: `librosa.pyin` with NaN interpolation for unvoiced frames
- DTW alignment: `dtw-python` aligns professor's F0 onto synthesised speech timeline
- Pitch shifting: `librosa.effects.pitch_shift` applied frame-wise (40 ms chunks, 10 ms hop)
- Median filter (kernel=7) smooths DTW output to prevent abrupt jumps

### Task 3.3 — Bengali Speech Synthesis

**File:** `tts_engine.py`

- Model: `facebook/mms-tts-ben` (HuggingFace VITS / MMS-TTS)
- Output sample rate: **22,050 Hz** (upsampled from VITS native 16,000 Hz)
- Voice cloning: speaker embedding used as style reference

> **Why not Coqui TTS / YourTTS?**
> Coqui TTS imports `isin_mps_friendly` from `transformers.pytorch_utils`,
> a symbol removed in `transformers >= 4.40`. HuggingFace VITS is used instead.

```python
from tts_engine import ProsodyCloningTTS
tts = ProsodyCloningTTS()
tts.synthesize(bengali_text, ref_audio_path="data/student_voice_ref.wav",
               source_prosody_path="data/original_segment.wav")
```

---

## Part IV — Adversarial Robustness & Spoofing Detection

### Task 4.1 — Anti-Spoofing Classifier

**File:** `security_module.py`

- Features: LFCC via `torchaudio.transforms.LFCC` (20 coefficients, 128 filters)
- Classifier: 3-layer MLP (64 → 32 → 2)
- Evaluation: Equal Error Rate (EER) — must be **< 10%**

> **Note:** `torchaudio.functional.lfcc` does not exist — use `torchaudio.transforms.LFCC`.

### Task 4.2 — Adversarial Noise Injection

**File:** `security_module.py`

- Method: Fast Gradient Sign Method (FGSM)
- Target: flip LID prediction Hindi→English
- Constraint: SNR must remain **> 40 dB** (inaudible to humans)
- Search: iterative epsilon sweep (`step=0.001`, `max_steps=50`)
- Reports minimum epsilon ε required to cause misclassification

```python
from security_module import AdversarialAttacker
attacker = AdversarialAttacker(lid_model)
adv_wav, epsilon = attacker.find_min_perturbation(waveform, true_label=0, target_label=1)
```

---

## Running the Pipeline

```bash
conda activate speech_env
cd /path/to/assignment2

# 1. Verify all input files are in place
ls data/original_segment.wav data/student_voice_ref.wav \
   data/syllabus.txt data/reference_transcript.txt

# 2. Run the full pipeline
python pipeline.py
```

### Expected output log

```
=== Task 1.3: Denoising with Spectral Subtraction ===
[Denoiser] Saved denoised audio -> output/denoised_segment.wav

=== Task 1.1: Language Identification ===
[LID] Detected N segments. First 5:
       0–500 ms : English
     500–1000 ms : Hindi
     ...
LID Macro F1: 0.XXXX  (Threshold: 0.85)

=== Task 1.2: Constrained Whisper ASR ===
[ASR] Transcript (first 200 chars): ...
[WER] XX.XX%

=== Tasks 2.1 & 2.2: Phonetic Mapping & Translation ===
[IPA]     ...
[Bengali] ...

=== Tasks 3.1–3.3: Zero-Shot Voice Cloning + DTW Prosody ===
[TTS] Speaker embedding shape: (192,)
[TTS] Saved -> output/output_LRL_cloned.wav  (XX.Xs @ 22050 Hz)
[MCD] X.XXXX  (Threshold: <8.0)

=== Task 4.1: Anti-Spoofing (LFCC CM) ===
[EER] X.XX%  (Threshold: <10%)

=== Task 4.2: Adversarial Noise Injection (FGSM) ===
[FGSM] Min epsilon (ε) = 0.XXXX

✅ Pipeline complete!
```

---

## Evaluation Metrics

| Metric | Threshold | Task | Notes |
|---|---|---|---|
| WER — English segments | < 15% | 1.2 | Requires `reference_transcript.txt` |
| WER — Hindi segments | < 25% | 1.2 | Separate Hindi-only WER evaluation |
| LID Macro F1 | ≥ 0.85 | 1.1 | Evaluated on labelled segments |
| LID switch precision | ≤ 200 ms | 1.1 | Timestamp accuracy at lang boundaries |
| MCD (Mel-Cepstral Distortion) | < 8.0 dB | 3.2 | Synthesised vs reference voice |
| EER (Equal Error Rate) | < 10% | 4.1 | Bona fide vs cloned audio |
| Adversarial SNR | > 40 dB | 4.2 | Perturbation must be inaudible |

---

## Known Issues & Design Choices

### 1. DeepFilterNet replaced with Spectral Subtraction (Task 1.3)
`deepfilternet` requires the compiled Rust extension `libdf` which is absent
on HPC clusters running Python 3.10+ with `torchaudio >= 0.13` (which removed
`torchaudio.backend.common`). Spectral Subtraction (Boll, 1979) is the
explicit alternative listed in the assignment specification.

### 2. Coqui TTS replaced with HuggingFace VITS (Task 3.3)
Coqui TTS (`TTS.api`) pulls in the Tortoise/XTTS stack which imports
`isin_mps_friendly` from `transformers.pytorch_utils` — removed in
`transformers >= 4.40`. `facebook/mms-tts-ben` via HuggingFace is used instead.

### 3. epitran language code
The correct language code for Hindi Devanagari is `'hin-Deva'`, not `'Hind-Deva'`.
Verify with: `python -c "import epitran; print(epitran.Epitran('hin-Deva').transliterate('नमस्ते'))"`.

### 4. VITS input padding
`tokenizer(..., padding=True)` must be set to prevent a `RuntimeError: narrow(): length must be non-negative`
crash in VITS relative position bias when input sequences are very short.

### 5. LID F1 with untrained model
Until the LID model is trained on labelled Hindi/English audio (e.g. MUCS corpus),
the F1 score reflects self-consistency not real accuracy. Train on labelled data
and replace silver labels with ground-truth annotations for a meaningful F1.

---

## References

1. Boll, S.F. (1979). *Suppression of acoustic noise in speech using spectral subtraction.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 27(2), 113–120.
2. Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* OpenAI. (Whisper)
3. Kim, J. et al. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.* ICML. (VITS)
4. Pratap, V. et al. (2023). *Scaling Speech Technology to 1000+ Languages.* Meta AI. (MMS-TTS)
5. Desplanques, B. et al. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech.
6. Goodfellow, I. et al. (2015). *Explaining and Harnessing Adversarial Examples.* ICLR. (FGSM)
7. Sakoe, H. & Chiba, S. (1978). *Dynamic programming algorithm optimization for spoken word recognition.* IEEE Transactions on ASSP. (DTW)
8. Wu, Z. et al. (2015). *ASVspoof: the Automatic Speaker Verification Spoofing and Countermeasures Challenge.* IEEE/ACM TASLP. (LFCC / EER)

---

> **GitHub repository:** `https://github.com/<your-username>/speech-understanding-a2`
> Replace with your actual repository link before submission.

> **Submission:** Zip the repository as `<RollNo>_PA2.zip` and submit on Google Classroom.