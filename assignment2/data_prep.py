"""
data_prep.py  —  Task 1.3: Denoising & Normalization
=====================================================
Uses Power-Spectrum Spectral Subtraction (Boll, 1979) implemented
in pure NumPy/librosa.  No deepfilternet / libdf / torchaudio.backend
dependency — works on any Python 3.8+ environment.

Reference:
  Boll, S.F. (1979). "Suppression of acoustic noise in speech using
  spectral subtraction." IEEE Trans. ASSP, 27(2), 113–120.
"""

import numpy as np
import soundfile as sf
import librosa
import torch
from config import SOURCE_AUDIO_PATH, DENOISED_PATH


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _estimate_noise_psd(stft_mag: np.ndarray,
                         noise_frames: int = 30) -> np.ndarray:
    """
    Estimate noise Power Spectral Density from the first
    `noise_frames` STFT frames (assumed to be noise/silence
    at lecture start).

    stft_mag : (n_freq, n_frames)  — magnitude spectrogram
    Returns  : (n_freq, 1)         — mean noise magnitude per bin
    """
    n = min(noise_frames, stft_mag.shape[1])
    return stft_mag[:, :n].mean(axis=1, keepdims=True)


def _rms_normalize(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """RMS-based loudness normalisation to target_db (LUFS approximate)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


# ──────────────────────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────────────────────

def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    over_subtraction_factor: float = 2.0,
    spectral_floor: float = 0.002,
    noise_frames: int = 30,
) -> np.ndarray:
    """
    Power-Spectrum Spectral Subtraction denoiser.

    Parameters
    ----------
    audio                   : 1-D float32 waveform
    sr                      : sample rate (Hz)
    n_fft                   : FFT window size
    hop_length              : hop between frames
    over_subtraction_factor : α — subtraction aggressiveness (1.0–2.5)
    spectral_floor          : β — prevents complete spectral nulling
    noise_frames            : number of initial frames used as noise estimate

    Returns
    -------
    denoised : np.ndarray, float32, same length as `audio`
    """
    # 1. Forward STFT
    stft  = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                          window='hann', center=True)
    mag   = np.abs(stft)        # (n_freq, n_frames)
    phase = np.angle(stft)

    # 2. Noise PSD estimate from first silent frames
    noise_est = _estimate_noise_psd(mag, noise_frames=noise_frames)

    # 3. Power-domain subtraction:  |Clean|² = |Noisy|² − α·|Noise|²
    clean_sq = mag**2 - over_subtraction_factor * noise_est**2

    # 4. Spectral floor to suppress musical noise artefacts
    floor    = (spectral_floor * noise_est) ** 2
    clean_sq = np.maximum(clean_sq, floor)

    # 5. Back to magnitude, resynthesize with original phase
    clean_stft = np.sqrt(clean_sq) * np.exp(1j * phase)
    denoised   = librosa.istft(clean_stft, hop_length=hop_length,
                                window='hann', center=True,
                                length=len(audio))
    return denoised.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Public API  (pipeline.py imports these two names)
# ──────────────────────────────────────────────────────────────

def denoise_with_deepfilternet(input_path: str, output_path: str,
                                target_sr: int = 16000) -> str:
    """
    Task 1.3 entry point — name kept for pipeline.py compatibility.

    Loads audio → Spectral Subtraction → RMS normalisation → saves.
    deepfilternet is NOT used: it requires torchaudio.backend.common
    which was removed in torchaudio ≥ 0.13, causing ImportError on
    modern envs.  Spectral Subtraction is the listed alternative in
    the assignment spec (§3, Task 1.3).

    Returns output_path.
    """
    print(f"[Denoiser] Loading  : {input_path}")
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    print(f"[Denoiser] Spectral Subtraction  (SR={sr} Hz, "
          f"duration={len(audio)/sr:.1f}s) ...")
    denoised = spectral_subtraction(audio, sr=sr)
    denoised = _rms_normalize(denoised, target_db=-23.0)

    sf.write(output_path, denoised, sr)
    print(f"[Denoiser] Saved denoised audio -> {output_path}")
    return output_path


def normalize_audio(waveform: torch.Tensor,
                     target_db: float = -23.0) -> torch.Tensor:
    """
    Torch-tensor RMS normalisation — used by other pipeline modules.
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-9:
        return waveform
    target_rms = 10 ** (target_db / 20.0)
    return waveform * (target_rms / rms)


# ──────────────────────────────────────────────────────────────
# Stand-alone test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    denoise_with_deepfilternet(SOURCE_AUDIO_PATH, DENOISED_PATH)
    print("✅ Denoising complete.")