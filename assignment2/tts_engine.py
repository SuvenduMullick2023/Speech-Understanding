# ============================================================
# CELL 10: tts_engine.py  (Tasks 3.1, 3.2, 3.3)
# FIX: DTW-warped F0 is now actually applied to the synthesized
#      waveform via PSOLA-style pitch shifting (using librosa),
#      not just printed as a debug message.
# ============================================================

import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from dtw import dtw as dtw_func
from TTS.api import TTS
from config import REF_VOICE_PATH, TARGET_LRL, OUTPUT_LRL_PATH


class ProsodyCloningTTS:
    """
    Task 3.1: Speaker embedding via YourTTS speaker_wav (d-vector internally).
    Task 3.2: DTW-based prosody warping of F0 from professor -> synthesised audio.
    Task 3.3: Final synthesis at ≥22.05 kHz using YourTTS / VITS.
    """

    TARGET_SR = 22050   # Assignment requires ≥22.05 kHz

    def __init__(self):
        print("[TTS] Loading YourTTS model...")
        # YourTTS: zero-shot cross-lingual voice cloning
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=True
        )

    # ------------------------------------------------------------------
    # Task 3.2: Prosody extraction
    # ------------------------------------------------------------------
    def extract_prosody(self, audio_path: str):
        """
        Returns interpolated F0 (no NaN) and energy contour.
        """
        y, sr = librosa.load(audio_path, sr=self.TARGET_SR)
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        energy = np.abs(librosa.stft(y)).mean(axis=0)

        # Interpolate NaN values (unvoiced frames)
        times       = np.arange(len(f0))
        good_idx    = ~np.isnan(f0)
        if good_idx.sum() > 1:
            f_interp = interp1d(
                times[good_idx], f0[good_idx],
                kind='linear', fill_value="extrapolate"
            )
            f0 = f_interp(times)
        else:
            f0 = np.zeros_like(f0)

        return f0, energy, sr

    # ------------------------------------------------------------------
    # Task 3.2: DTW warping
    # ------------------------------------------------------------------
    def apply_dtw_warping(self, source_f0: np.ndarray, target_f0: np.ndarray) -> np.ndarray:
        """
        Align source (professor) F0 onto target (synthesised) length using DTW.
        Returns warped_f0 of same length as target_f0.

        FIX: Previously only printed a message; now returns the warped array
             which is used to pitch-shift the synthesised waveform.
        """
        alignment  = dtw_func(source_f0, target_f0, keep_internals=True)
        idx_source = alignment.index1
        idx_target = alignment.index2

        warped_f0 = np.zeros_like(target_f0)
        for s_i, t_i in zip(idx_source, idx_target):
            if t_i < len(warped_f0):
                warped_f0[t_i] = source_f0[s_i]

        # Smooth with a small median filter to avoid sharp jumps
        from scipy.signal import medfilt
        warped_f0 = medfilt(warped_f0, kernel_size=7)
        return warped_f0

    def _apply_pitch_shift_to_waveform(
        self, waveform: np.ndarray, sr: int,
        target_f0: np.ndarray, warped_f0: np.ndarray
    ) -> np.ndarray:
        """
        Apply frame-wise pitch shifting so synthesised F0 matches warped_f0.
        Uses librosa effects.pitch_shift on short overlapping chunks.
        """
        hop = librosa.time_to_samples(0.010, sr=sr)  # 10 ms hop
        out = np.zeros_like(waveform)
        n_frames = min(len(target_f0), len(warped_f0))

        for i in range(n_frames):
            t0 = i * hop
            t1 = min(t0 + hop * 4, len(waveform))   # ~40 ms chunk
            chunk = waveform[t0:t1]

            tgt = target_f0[i]
            war = warped_f0[i]

            if tgt > 0 and war > 0 and tgt != war:
                n_steps = 12.0 * np.log2(war / tgt)  # semitones
                chunk   = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=n_steps)

            # Overlap-add
            out_end = min(t0 + len(chunk), len(out))
            out[t0:out_end] += chunk[:out_end - t0]

        return out

    # ------------------------------------------------------------------
    # Task 3.3: Synthesis
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        ref_audio_path: str = REF_VOICE_PATH,
        source_prosody_path: str = None,
        output_path: str = OUTPUT_LRL_PATH
    ):
        """
        1. Synthesise speech in TARGET_LRL using student voice reference.
        2. If source_prosody_path provided, apply DTW F0 warping.
        3. Save output at ≥22.05 kHz.
        """
        print(f"[TTS] Synthesising Bengali text ({len(text)} chars)...")
        wav = self.tts.tts(
            text=text,
            speaker_wav=ref_audio_path,
            language=TARGET_LRL
        )
        wav = np.array(wav, dtype=np.float32)

        if source_prosody_path is not None:
            print("[TTS] Extracting prosody for DTW warping...")
            src_f0, _, _   = self.extract_prosody(source_prosody_path)
            tgt_f0, _, sr  = self.extract_prosody(ref_audio_path)
            warped_f0      = self.apply_dtw_warping(src_f0, tgt_f0)

            # Resample wav to TARGET_SR if needed
            if sr != self.TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.TARGET_SR)

            # FIX: actually apply pitch warping to the waveform
            print("[TTS] Applying DTW pitch warp to synthesised audio...")
            wav = self._apply_pitch_shift_to_waveform(
                wav, self.TARGET_SR, tgt_f0, warped_f0
            )

        sf.write(output_path, wav, self.TARGET_SR)
        print(f"[TTS] Saved synthesised audio -> {output_path}  (SR={self.TARGET_SR} Hz)")
        return wav, self.TARGET_SR