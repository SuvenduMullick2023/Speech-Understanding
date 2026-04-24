# ============================================================
# CELL 12: pipeline.py  (Main Orchestrator)
# FIX: Added denoising step (Task 1.3), LID timestamp output,
#      WER/MCD/EER evaluation stubs, and correct zip naming.
# ============================================================
import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer

from config import *
from data_prep   import denoise_with_deepfilternet, normalize_audio
from lid_model   import FrameLevelLID
from asr_engine  import ConstrainedWhisperASR
from translation_module import PhoneticTranslator
from tts_engine  import ProsodyCloningTTS
from security_module import LFCCSpoofDetector, AdversarialAttacker, compute_eer


def compute_mcd(ref_wav_path: str, syn_wav_path: str) -> float:
    """
    Mel-Cepstral Distortion between reference voice and synthesised output.
    Must be < 8.0 (assignment §5).
    """
    import librosa
    ref, sr1 = librosa.load(ref_wav_path, sr=22050)
    syn, sr2 = librosa.load(syn_wav_path, sr=22050)

    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr1, n_mfcc=13)
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr2, n_mfcc=13)

    min_len  = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    diff     = ref_mfcc[:, :min_len] - syn_mfcc[:, :min_len]
    mcd      = (10.0 / np.log(10)) * np.sqrt(2.0 * np.mean(diff ** 2))
    print(f"[MCD] {mcd:.4f}  (Threshold: <{MAX_MCD})")
    return mcd


def main():
    
    import os
    PROJECT_PATH = "/home/suvendu/speech_understanding/Speech-Understanding/assignment2"
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_PATH, "data"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_PATH, "output"), exist_ok=True)
    os.chdir(PROJECT_PATH)
    print("✅ Environment Ready. Working Directory:", os.getcwd())
    ls -l
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Using device: {device}")

    # ── Validate audio files exist ───────────────────────────────────
    for path, name in [(SOURCE_AUDIO_PATH, "original_segment.wav"),
                       (REF_VOICE_PATH,    "student_voice_ref.wav")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Please upload '{name}' to your Drive data folder."
            )

    # ── Task 1.3: Denoising ──────────────────────────────────────────
    print("\n=== Task 1.3: Denoising with DeepFilterNet ===")
    denoised_path = denoise_with_deepfilternet(SOURCE_AUDIO_PATH, DENOISED_PATH)

    # ── Task 1.1: LID Model (initialise; train separately or load weights) ──
    print("\n=== Task 1.1: Language Identification ===")
    lid_model = FrameLevelLID().to(device)
    # NOTE: In a full submission, load pre-trained weights here:
    #   lid_model.load_state_dict(torch.load("lid_weights.pt", map_location=device))
    lid_model.eval()

    waveform, sr = torchaudio.load(denoised_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.to(device)

    lid_segments = lid_model.predict_language(waveform.squeeze(0), device)
    print(f"[LID] Detected {len(lid_segments)} segments. First 5:")
    for seg in lid_segments[:5]:
        print(f"  {seg['start_ms']:6d}–{seg['end_ms']:6d} ms : {seg['lang_label']}")

    # ── Task 1.2: ASR with Constrained Decoding ──────────────────────
    print("\n=== Task 1.2: Constrained Whisper ASR ===")
    asr = ConstrainedWhisperASR()
    audio_np = waveform.squeeze(0).cpu().numpy()
    hinglish_text = asr.transcribe_with_bias(audio_np, language="hi")
    print(f"[ASR] Transcript (first 200 chars): {hinglish_text[:200]}...")

    # WER evaluation (requires reference transcript file)
    ref_transcript_path = os.path.join(DATA_DIR, "reference_transcript.txt")
    if os.path.exists(ref_transcript_path):
        with open(ref_transcript_path, "r") as f:
            ref_text = f.read().strip()
        word_error = compute_wer(ref_text, hinglish_text)
        print(f"[WER] {word_error*100:.2f}%")
    else:
        print("[WER] Skipped: no reference transcript found at", ref_transcript_path)

    # ── Tasks 2.1 & 2.2: IPA + Translation ───────────────────────────
    print("\n=== Tasks 2.1 & 2.2: Phonetic Mapping & Translation ===")
    translator   = PhoneticTranslator()
    ipa_string   = translator.hinglish_to_ipa(hinglish_text)
    bengali_text = translator.translate_with_glossary(hinglish_text)
    print(f"[IPA]     {ipa_string[:100]}...")
    print(f"[Bengali] {bengali_text[:100]}...")

    # ── Tasks 3.1–3.3: TTS with DTW Prosody Warping ──────────────────
    print("\n=== Tasks 3.1–3.3: Zero-Shot Voice Cloning + DTW Prosody ===")
    tts_engine = ProsodyCloningTTS()
    syn_wav, syn_sr = tts_engine.synthesize(
        text=bengali_text,
        ref_audio_path=REF_VOICE_PATH,
        source_prosody_path=SOURCE_AUDIO_PATH,
        output_path=OUTPUT_LRL_PATH
    )
    print(f"[TTS] Final audio: {OUTPUT_LRL_PATH}  ({len(syn_wav)/syn_sr:.1f}s @ {syn_sr}Hz)")

    # MCD evaluation
    mcd_score = compute_mcd(REF_VOICE_PATH, OUTPUT_LRL_PATH)
    assert mcd_score < MAX_MCD, f"MCD {mcd_score:.2f} exceeds threshold {MAX_MCD}"

    # ── Tasks 4.1 & 4.2: Anti-Spoofing + Adversarial Attack ─────────
    print("\n=== Task 4.1: Anti-Spoofing (LFCC CM) ===")
    spoof_detector = LFCCSpoofDetector().to(device)
    # Demo EER with dummy scores (replace with real inference in full submission)
    bf_scores    = np.random.beta(2, 5, 100)   # bona fide: low spoof score
    spoof_scores = np.random.beta(5, 2, 100)   # spoof:     high spoof score
    eer_score    = compute_eer(bf_scores, spoof_scores)
    print(f"[EER] Demo EER = {eer_score*100:.2f}%  (replace with real eval data)")

    print("\n=== Task 4.2: Adversarial Noise Injection (FGSM) ===")
    attacker     = AdversarialAttacker(lid_model)
    wf_5s        = waveform[..., :16000 * 5]   # 5-second segment
    adv_wav, eps = attacker.find_min_perturbation(wf_5s, true_label=0, target_label=1)

    if eps is not None:
        print(f"[FGSM] Min epsilon (ε) = {eps:.4f}")
    else:
        print("[FGSM] No successful adversarial example found within SNR constraints.")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()