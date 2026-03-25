

q1_cepstral_pipeline/
├── mfcc_manual.py              # Handcrafted MFCC/cepstrum engine
├── leakage_snr.py              # Spectral leakage & SNR analysis
├── voiced_unvoiced.py          # Boundary detection algorithm
├── phonetic_mapping.py         # Wav2Vec2 alignment + RMSE
├── q1_report.pdf               # This report (4 pages max)
├── requirements.txt            # Python dependencies
├── data/
│   ├── manifest.txt            # Audio file listing
│   └── voice_speech_Q1_1.m4a   # Test audio file (29.98s)
└── outputs/
    ├── mfcc_plots.pdf          #  MFCC extraction visualization (real speech)
    ├── leakage_comparison.pdf  # Window function comparison
    ├── boundary_detection.pdf  # Voiced/unvoiced detection
    ├── alignment_results.pdf   # Phone alignment visualization
    ├── alignment_results.json  # RMSE: 27.22ms (short segment)
    ├── phone_report.md         #  Phone-level details: RMSE 57.21ms (full audio)
    └── leakage_results.json    # Window analysis metrics





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
- **License**: CC BY 4.0

## Installation
```bash
pip install -r requirements.txt

