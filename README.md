

# Q1 Cepstral Pipeline: Speech Analysis & Alignment 

A comprehensive signal processing pipeline for MFCC extraction, spectral leakage analysis, voiced/unvoiced boundary detection, and phonetic alignment using Wav2Vec2.

# File Structure

```text
q1_cepstral_pipeline/
├── mfcc_manual.py          # Handcrafted MFCC and cepstrum extraction engine
├── leakage_snr.py          # Analysis of spectral leakage and SNR metrics
├── voiced_unvoiced.py      # Algorithm for speech boundary detection
├── phonetic_mapping.py     # Wav2Vec2 alignment and RMSE calculation
├── q1_report.pdf           # Technical project report (Max 4 pages)
├── requirements.txt        # Python dependencies (Torch, Librosa, etc.)
├── data/
│   ├── manifest.txt        # Audio file listing and metadata
│   └── voice_speech_Q1_1.m4a # Primary test audio file (29.98s)
└── outputs/
    ├── mfcc_plots.pdf      # Visualizations of MFCC extraction
    ├── leakage_comparison.pdf # Window function performance comparison
    ├── boundary_detection.pdf # Voiced vs. Unvoiced detection results
    ├── alignment_results.pdf  # Phonetic alignment visualization
    ├── alignment_results.json # Segment RMSE: 27.22ms
    ├── phone_report.md     # Full audio detail: Total RMSE 57.21ms
    └── leakage_results.json   # Quantitative window analysis metrics

Assignment1_1_speech_understanding_v1.ipynb file is the main file for generating for all result

# Q3 Ethical Audio: Privacy & Fairness Pipeline

This repository contains a framework for ethical speech processing, focusing on adaptive privacy obfuscation, fairness auditing across demographic cohorts, and privacy-utility tradeoff analysis.

##  File Structure

```text
q3_ethical_audio/
├── audit.py                # Pipeline for auditing model bias and fairness
├── privacymodule.py        # Core logic (Updated with AdaptivePrivacyObfuscator)
├── train_fair.py           # Training script with fairness-aware constraints
├── pp_demo.py              # Privacy-Preserving (PP) demonstration script
├── q3_report.md            # Summary of ethical findings and methodology
├── audit_plots.pdf         # Visualizations of bias and fairness metrics
├── training_curves.pdf     # Loss and accuracy curves (Privacy vs. Utility)
├── evaluation_scripts/     # Advanced metric suites
│   ├── metrics.py          # Standard performance metrics (WER, Accuracy)
│   ├── fairness_metrics.py # NEW: compute_fairness_metrics (EO, DP)
│   ├── privacy_utility_analysis.py # NEW: analyze_privacy_utility_tradeoff
│   └── visualize_audio.py  # NEW: visualize_audio_comparison (Spectrograms)
├── examples/               # Audio samples for verification
│   ├── original_0.wav      # Raw audio input
│   ├── obfuscated_0.wav    # Audio after Adaptive Privacy Obfuscation
│   └── ...
└── librispeech_data/       # Dataset directory (LibriSpeech subset)

Assignment1_3_speech_understanding.ipynb  file is the main file for generating for all result







# Speech-Understanding

# Ethical Auditing & Privacy-Preserving AI Report (Enhanced)

## 1. Documentation Debt Analysis
- **Dataset**: LibriSpeech ASR (clean subset)
- **Debt Score**: 0.75 (75% metadata missing)
- **Missing Fields**: gender, age, accent
- **Impact**: Cannot audit fairness without demographic metadata

## 2. Privacy Module (Adaptive)
- **Method**: Adaptive pitch-shifting (configurable privacy level)
- **Optimal Shift**: 2.0 semitones (balanced privacy-utility)
- **Privacy Effective**: Yes (pitch shift detected >100Hz)

## 3. Privacy-Utility Trade-off
- **Analysis**: Tested shifts from 0.5 to 4.0 semitones
- **Recommended**: 2.0 semitones (STOI > 0.7, privacy score > 0.5)
- **Previous Issue**: 4.0 semitones caused quality degradation

## 4. Fairness Training
- **Technique**: Gradient Reversal Layer
- **Result**: Gender accuracy reduced to 0.500 (random chance)
- **Fairness Metrics**: Demographic Parity Difference < 0.1

## 5. Audio Quality
- **PESQ Proxy**: 3.0+ (with adaptive shift)
- **STOI Proxy**: 0.7+ (intelligibility preserved)
- **Spectral Difference**: Visualized in audio_comparison.pdf

## 6. Ethical Considerations
-  Privacy: Biometric traits obfuscated
-  Fairness: Demographic gaps minimized
-  Transparency: Trade-offs documented
-  Limitation: Synthetic demographics (LibriSpeech lacks metadata)

## 7. Recommendations
1. Use Common Voice for real demographic metadata
2. Apply adaptive privacy (2.0 semitones) for production
3. Run regular fairness audits with compute_fairness_metrics()
4. Monitor privacy-utility trade-off continuously

## 8. Conclusion
The enhanced pipeline successfully balances privacy and utility 
through adaptive obfuscation. Fairness training reduces demographic 

## Dataset
- **Name**: LibriSpeech ASR (clean subset)
- **Samples**: 500 audio files
-



