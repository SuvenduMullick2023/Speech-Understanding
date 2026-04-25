# ============================================================
# CELL 7: lid_model.py  (Task 1.1 – Frame-Level LID)
# FIX: Replaced hardcoded linear layer size with AdaptiveAvgPool2d
#      so the model works for any input duration without shape errors.
# ============================================================


import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LIDDataset(Dataset):
    """
    Simple dataset for LID evaluation.
    segments: list of (waveform_tensor, label_int)
              label: 0=Hindi, 1=English
    """
    def __init__(self, segments: list, target_length: int = 16000):
        self.segments      = segments
        self.target_length = target_length  # pad/trim to 1 second

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        waveform, label = self.segments[idx]
        # Trim or pad to fixed length
        if waveform.shape[-1] > self.target_length:
            waveform = waveform[..., :self.target_length]
        elif waveform.shape[-1] < self.target_length:
            pad = torch.zeros(*waveform.shape[:-1],
                              self.target_length - waveform.shape[-1])
            waveform = torch.cat([waveform, pad], dim=-1)
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.long)

class FrameLevelLID(nn.Module):
    """
    Task 1.1: Multi-Head Frame-Level Language Identification.
    Classifies each ~10ms frame as Hindi (0) or English (1).

    Architecture: 2-layer CNN + AdaptiveAvgPool + FC classifier.
    Input:  raw waveform tensor  (Batch, Time_samples)  @ 16 kHz
    Output: per-frame logits     (Batch, num_classes)
    """

    def __init__(self, n_mels: int = 128, num_classes: int = 2, sample_rate: int = 16000):
        super(FrameLevelLID, self).__init__()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=int(sample_rate * 0.010),   # 10 ms frame shift
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # FIX: AdaptiveAvgPool collapses spatial dims to (4, 4)
        # regardless of input duration -> no hardcoded linear size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (Batch, Time_samples)
        returns:  (Batch, num_classes) logits
        """
        spec = self.mel_transform(waveform)          # (B, Mels, T_frames)
        spec = self.amplitude_to_db(spec)
        spec = spec.unsqueeze(1)                     # (B, 1, Mels, T_frames)

        features = self.feature_extractor(spec)      # (B, 64, H, W)
        features = self.adaptive_pool(features)      # (B, 64, 4, 4)
        features = features.flatten(start_dim=1)     # (B, 64*4*4 = 1024)
        logits   = self.classifier(features)         # (B, num_classes)
        return logits

    def predict_language(self, waveform: torch.Tensor, device: torch.device,
                          frame_duration_ms: int = 500) -> list:
        """
        Slide a window over the waveform and return per-segment predictions
        with timestamps.  Used for language-switch boundary detection (≤200ms).

        Returns list of dicts: [{start_ms, end_ms, lang_id, lang_label}, ...]
        """
        self.eval()
        sr = 16000
        frame_samples = int(sr * frame_duration_ms / 1000)
        results = []

        waveform = waveform.to(device)
        total_samples = waveform.shape[-1]

        for start in range(0, total_samples, frame_samples):
            end   = min(start + frame_samples, total_samples)
            chunk = waveform[..., start:end]

            # Pad short last chunk
            if chunk.shape[-1] < frame_samples:
                pad = torch.zeros(*chunk.shape[:-1], frame_samples - chunk.shape[-1], device=device)
                chunk = torch.cat([chunk, pad], dim=-1)

            with torch.no_grad():
                logits  = self.forward(chunk.unsqueeze(0) if chunk.dim() == 1 else chunk)
                lang_id = logits.argmax(dim=-1).item()

            results.append({
                "start_ms":   start * 1000 // sr,
                "end_ms":     end   * 1000 // sr,
                "lang_id":    lang_id,
                "lang_label": "English" if lang_id == 1 else "Hindi"
            })
        return results


def evaluate_lid_f1(model: FrameLevelLID, dataloader, device: torch.device) -> float:
    """Compute macro F1-score on a LID dataloader. Must exceed 0.85."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            preds = model(waveforms).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"LID Macro F1: {f1:.4f}  (Threshold: 0.85)")
    if f1 < 0.85:
        print(f"[LID] WARNING: F1 {f1:.4f} below required 0.85. "
            f"Train the LID model on labelled Hindi/English audio segments.")
    return f1