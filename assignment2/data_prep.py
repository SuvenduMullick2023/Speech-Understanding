# ============================================================
# CELL 6: data_prep.py  (Task 1.3 – Denoising & Normalization)
# FIX: This file was COMPLETELY MISSING from the original code.
#      DeepFilterNet is used as required by the assignment.
# ============================================================

import torch
import torchaudio
import soundfile as sf
import numpy as np
from df.enhance import enhance, init_df, load_audio, save_audio
from config import SOURCE_AUDIO_PATH, DENOISED_PATH


def denoise_with_deepfilternet(input_path: str, output_path: str) -> str:
    """
    Task 1.3: Denoising & Normalization using DeepFilterNet.
    Removes classroom background noise and reverb.

    Args:
        input_path:  Path to raw lecture audio (.wav, any SR).
        output_path: Where to save the denoised audio.

    Returns:
        output_path after successful denoising.
    """
    print(f"[DeepFilterNet] Loading model...")
    model, df_state, _ = init_df()   # Downloads DF2 weights on first run

    print(f"[DeepFilterNet] Loading audio: {input_path}")
    audio, _ = load_audio(input_path, sr=df_state.sr())

    print(f"[DeepFilterNet] Enhancing (this may take a moment for 10-min audio)...")
    enhanced = enhance(model, df_state, audio)

    save_audio(output_path, enhanced, df_state.sr())
    print(f"[DeepFilterNet] Saved denoised audio -> {output_path}")
    return output_path

def normalize_audio(waveform: torch.Tensor, target_db: float = -23.0) -> torch.Tensor:
    """
    RMS-based loudness normalization to target_db LUFS (approximate).
    Runs after denoising to ensure consistent input levels for ASR/LID.
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms == 0:
        return waveform
    target_rms = 10 ** (target_db / 20.0)
    return waveform * (target_rms / rms)


if __name__ == "__main__":
    denoise_with_deepfilternet(SOURCE_AUDIO_PATH, DENOISED_PATH)
    print("✅ Denoising complete.")
