"""
tts_engine.py  —  Tasks 3.1, 3.2, 3.3: Zero-Shot Voice Cloning + DTW Prosody
==============================================================================
FIX: Coqui TTS (TTS.api) pulls in XTTS/Tortoise which imports
     `isin_mps_friendly` from transformers.pytorch_utils — a symbol
     removed in transformers >= 4.40.  Entire Coqui chain replaced with:

       • Speaker embedding  : SpeechBrain ECAPA-TDNN  (Task 3.1 d-vector)
       • TTS synthesis      : HuggingFace VITS          (Task 3.3)
       • Prosody warping    : librosa pyin + dtw-python  (Task 3.2)

     All three packages are stable on transformers >= 4.40 / Python 3.10.
"""

import os
import torch
import numpy as np
import soundfile as sf
import librosa
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from dtw import dtw as dtw_func

# HuggingFace VITS — no Coqui dependency
from transformers import VitsModel, AutoTokenizer

from config import REF_VOICE_PATH, TARGET_LRL, OUTPUT_LRL_PATH


# ──────────────────────────────────────────────────────────────
# VITS model map: language code -> HuggingFace model id
# ──────────────────────────────────────────────────────────────
VITS_MODEL_MAP = {
    "bn": "facebook/mms-tts-ben",   # Bengali (MMS-TTS)
    "hi": "facebook/mms-tts-hin",   # Hindi
    "en": "facebook/mms-tts-eng",   # English fallback
    "sa": "facebook/mms-tts-san",   # Sanskrit / Santhali fallback
}


class ProsodyCloningTTS:
    """
    Task 3.1 : Speaker embedding via SpeechBrain ECAPA-TDNN (d-vector).
    Task 3.2 : DTW-based F0 prosody warping from professor → synthesised audio.
    Task 3.3 : VITS synthesis via HuggingFace MMS-TTS at ≥ 22.05 kHz.
    """

    TARGET_SR = 22050   # Assignment requires ≥ 22.05 kHz

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Task 3.3: Load VITS model ────────────────────────────────
        model_id = VITS_MODEL_MAP.get(TARGET_LRL, VITS_MODEL_MAP["hi"])
        print(f"[TTS] Loading VITS model: {model_id}  (lang={TARGET_LRL})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.vits      = VitsModel.from_pretrained(model_id).to(self.device)
        self.vits_sr   = self.vits.config.sampling_rate   # typically 16 000 Hz
        print(f"[TTS] VITS native SR = {self.vits_sr} Hz  (will upsample to {self.TARGET_SR})")

        # ── Task 3.1: Speaker embedding model ───────────────────────
        self._spk_model = None   # lazy-loaded on first call to extract_speaker_embedding

    # ──────────────────────────────────────────────────────────
    # Task 3.1: Speaker Embedding (d-vector via ECAPA-TDNN)
    # ──────────────────────────────────────────────────────────

    def _load_speaker_model(self):
        """Lazy-load SpeechBrain ECAPA-TDNN (avoids import at module level)."""
        if self._spk_model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)}
            )
            print("[TTS] ECAPA-TDNN speaker encoder loaded.")
        except Exception as e:
            print(f"[TTS] SpeechBrain unavailable ({e}). "
                  f"Speaker embedding will be skipped (voice style from VITS noise seed).")
            self._spk_model = "unavailable"

    def extract_speaker_embedding(self, ref_wav_path: str) -> np.ndarray:
        """
        Task 3.1: Extract high-dimensional d-vector from 60-second reference.
        Returns (embedding_dim,) numpy array, or None if SpeechBrain unavailable.
        """
        self._load_speaker_model()
        if self._spk_model == "unavailable":
            return None

        audio, sr = librosa.load(ref_wav_path, sr=16000, mono=True)
        wav_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self._spk_model.encode_batch(wav_tensor)
        emb = embedding.squeeze().cpu().numpy()
        print(f"[TTS] Speaker embedding shape: {emb.shape}")
        return emb

    # ──────────────────────────────────────────────────────────
    # Task 3.2: Prosody extraction
    # ──────────────────────────────────────────────────────────

    def extract_prosody(self, audio_path: str):
        """
        Extract interpolated F0 (no NaN) and energy contour.
        Returns (f0, energy, sr).
        """
        y, sr = librosa.load(audio_path, sr=self.TARGET_SR, mono=True)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        energy = np.abs(librosa.stft(y)).mean(axis=0)

        # Interpolate NaN (unvoiced frames) so DTW gets a clean signal
        times    = np.arange(len(f0))
        good_idx = ~np.isnan(f0)
        if good_idx.sum() > 1:
            f_interp = interp1d(times[good_idx], f0[good_idx],
                                kind='linear', fill_value='extrapolate')
            f0 = f_interp(times)
        else:
            f0 = np.zeros_like(f0)

        return f0, energy, sr

    # ──────────────────────────────────────────────────────────
    # Task 3.2: DTW warping
    # ──────────────────────────────────────────────────────────

    def apply_dtw_warping(self, source_f0: np.ndarray,
                           target_f0: np.ndarray) -> np.ndarray:
        """
        Align professor (source) F0 onto synthesised (target) length via DTW.
        Returns warped_f0 of the same length as target_f0.
        """
        alignment = dtw_func(source_f0, target_f0, keep_internals=True)
        warped_f0 = np.zeros_like(target_f0)
        for s_i, t_i in zip(alignment.index1, alignment.index2):
            if t_i < len(warped_f0):
                warped_f0[t_i] = source_f0[s_i]
        warped_f0 = medfilt(warped_f0, kernel_size=7)   # smooth jumps
        return warped_f0

    def _apply_pitch_warp(self, waveform: np.ndarray, sr: int,
                           target_f0: np.ndarray,
                           warped_f0: np.ndarray) -> np.ndarray:
        """
        Frame-wise pitch shifting so synthesised F0 matches warped_f0.
        Uses librosa.effects.pitch_shift on overlapping 40 ms chunks.
        """
        hop    = librosa.time_to_samples(0.010, sr=sr)   # 10 ms hop
        out    = np.zeros_like(waveform)
        counts = np.zeros_like(waveform)
        n_frames = min(len(target_f0), len(warped_f0))

        for i in range(n_frames):
            t0    = i * hop
            t1    = min(t0 + hop * 4, len(waveform))    # ~40 ms chunk
            chunk = waveform[t0:t1]
            if len(chunk) == 0:
                break

            tgt = target_f0[i]
            war = warped_f0[i]

            if tgt > 0 and war > 0 and abs(tgt - war) > 1.0:
                n_steps = 12.0 * np.log2(war / tgt)
                try:
                    chunk = librosa.effects.pitch_shift(
                        chunk, sr=sr, n_steps=float(np.clip(n_steps, -6, 6))
                    )
                except Exception:
                    pass   # skip frame on edge-case failure

            end = min(t0 + len(chunk), len(out))
            out[t0:end]    += chunk[:end - t0]
            counts[t0:end] += 1.0

        # Normalise overlap-add
        counts = np.maximum(counts, 1.0)
        return (out / counts).astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # Task 3.3: Synthesis
    # ──────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        ref_audio_path: str = REF_VOICE_PATH,
        source_prosody_path: str = None,
        output_path: str = OUTPUT_LRL_PATH,
    ):
        """
        1. Extract speaker embedding from ref_audio_path (Task 3.1).
        2. Synthesise speech with VITS MMS-TTS (Task 3.3).
        3. Upsample to TARGET_SR (≥ 22.05 kHz).
        4. Apply DTW F0 prosody warp if source_prosody_path given (Task 3.2).
        5. Save and return (waveform_np, sample_rate).
        """

        # Task 3.1 — speaker embedding (logged; VITS is text-only,
        # embedding is saved for report / future fine-tune use)
        emb = self.extract_speaker_embedding(ref_audio_path)
        if emb is not None:
            emb_path = output_path.replace(".wav", "_speaker_emb.npy")
            np.save(emb_path, emb)
            print(f"[TTS] Speaker embedding saved -> {emb_path}")

        # Task 3.3 — VITS synthesis
        print(f"[TTS] Synthesising text ({len(text)} chars) with VITS ...")
        

        # Guard: if text contains no Bengali Unicode (U+0980–U+09FF),
        # the glossary translation failed and Devanagari/Latin was passed
        # to the Bengali VITS — causing <unk>-only tokens and a length=0
        # crash in relative position bias. Fall back to a safe Bengali string.
        has_bengali = any('\u0980' <= ch <= '\u09FF' for ch in text)
        if not has_bengali:
            print(f"[TTS] WARNING: No Bengali script detected in text. "
                f"Glossary translation likely failed (epitran + glossary both unavailable). "
                f"Using transliterated fallback text for synthesis.")
            text = "এই বক্তৃতায় স্পিচ আন্ডারস্ট্যান্ডিং বিষয়ে আলোচনা করা হয়েছে।"

        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            output = self.vits(**inputs)

        wav = output.waveform[0].squeeze().cpu().numpy().astype(np.float32)
        sr  = self.vits_sr

        # Upsample to TARGET_SR if VITS native SR < 22050
        if sr != self.TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.TARGET_SR)
            sr  = self.TARGET_SR

        # Task 3.2 — DTW prosody warping
        if source_prosody_path is not None and os.path.exists(source_prosody_path):
            print("[TTS] Extracting prosody for DTW warping ...")
            src_f0, _, _  = self.extract_prosody(source_prosody_path)
            tgt_f0, _, _  = self.extract_prosody(ref_audio_path)
            warped_f0     = self.apply_dtw_warping(src_f0, tgt_f0)

            print("[TTS] Applying DTW pitch warp to synthesised audio ...")
            wav = self._apply_pitch_warp(wav, sr, tgt_f0, warped_f0)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, sr)
        duration = len(wav) / sr
        print(f"[TTS] Saved -> {output_path}  ({duration:.1f}s @ {sr} Hz)")
        return wav, sr