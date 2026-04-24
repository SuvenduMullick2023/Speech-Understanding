# ============================================================
# CELL 11: security_module.py  (Tasks 4.1 & 4.2)
# FIX: torchaudio.functional.lfcc does not exist.
#      Replaced with torchaudio.transforms.LFCC (correct API).
# ============================================================

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from sklearn.metrics import roc_curve
from config import FGSM_EPSILON_STEP, MAX_FGSM_STEPS, SNR_THRESHOLD_DB


# ------------------------------------------------------------------
# Task 4.1: Anti-Spoofing Classifier (CM) using LFCC
# ------------------------------------------------------------------
class LFCCSpoofDetector(nn.Module):
    """
    Countermeasure (CM) system based on LFCC features.
    Classifies audio as Bona Fide (0) or Spoof (1).
    Evaluated using Equal Error Rate (EER).
    """

    def __init__(self, n_lfcc: int = 20, sample_rate: int = 16000):
        super(LFCCSpoofDetector, self).__init__()
        self.n_lfcc      = n_lfcc
        # FIX: use torchaudio.transforms.LFCC, not torchaudio.functional.lfcc
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=128,
            n_lfcc=n_lfcc,
            speckwargs={"n_fft": 512, "hop_length": 160, "center": False}
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_lfcc, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # 0=BF, 1=Spoof
        )

    def extract_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (Batch, 1, Time) or (Batch, Time)
        Returns mean LFCC over time: (Batch, n_lfcc)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)   # -> (B, 1, T)
        lfcc = self.lfcc_transform(waveform)   # (B, 1, n_lfcc, T_frames)
        return lfcc.squeeze(1).mean(dim=-1)    # (B, n_lfcc)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.extract_lfcc(waveform)
        return self.classifier(features)


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    bonafide_scores: model spoof-probability scores for real audio (label=0)
    spoof_scores:    model spoof-probability scores for synthesised audio (label=1)
    Must achieve EER < 10%.
    """
    labels = np.concatenate([
        np.zeros(len(bonafide_scores)),
        np.ones(len(spoof_scores))
    ])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # EER is where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer     = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))
    print(f"[EER] {eer * 100:.2f}%  (Threshold: <10%)")
    return eer


# ------------------------------------------------------------------
# Task 4.2: Adversarial Noise Injection via FGSM
# ------------------------------------------------------------------
class AdversarialAttacker:
    """
    FGSM attack on the LID model.
    Goal: find minimum epsilon that causes LID to misclassify Hindi->English
          while keeping SNR > 40 dB (inaudible perturbation).
    """

    def __init__(self, lid_model: nn.Module):
        self.lid_model = lid_model
        self.lid_model.eval()

    def fgsm_attack(
        self, waveform: torch.Tensor, true_label: int, epsilon: float
    ) -> torch.Tensor:
        waveform = waveform.clone().detach().requires_grad_(True)
        outputs = self.lid_model(waveform)
        loss    = nn.CrossEntropyLoss()(outputs, torch.tensor([true_label], device=waveform.device))

        self.lid_model.zero_grad()
        loss.backward()

        data_grad          = waveform.grad.data.sign()
        perturbed_waveform = waveform + epsilon * data_grad
        perturbed_waveform = torch.clamp(perturbed_waveform, -1.0, 1.0)
        return perturbed_waveform.detach()

    def compute_snr(self, original: torch.Tensor, perturbed: torch.Tensor) -> float:
        noise        = perturbed - original
        sig_power    = torch.mean(original ** 2)
        noise_power  = torch.mean(noise    ** 2)
        if noise_power == 0:
            return float('inf')
        return 10.0 * torch.log10(sig_power / noise_power).item()

    def find_min_perturbation(
        self, waveform: torch.Tensor,
        true_label: int, target_label: int
    ):
        """
        Iteratively increase epsilon until:
          (a) LID prediction flips to target_label, AND
          (b) SNR remains > SNR_THRESHOLD_DB (40 dB).

        Returns: (adversarial_waveform, epsilon) or (original, None) on failure.
        """
        current_eps = FGSM_EPSILON_STEP

        for step in range(MAX_FGSM_STEPS):
            adv_wav = self.fgsm_attack(waveform, true_label, current_eps)

            with torch.no_grad():
                pred = self.lid_model(adv_wav).argmax(dim=1).item()

            snr_db = self.compute_snr(waveform.detach(), adv_wav)

            if pred == target_label and snr_db > SNR_THRESHOLD_DB:
                print(f"[FGSM] ✅ Flip at ε={current_eps:.4f}, SNR={snr_db:.2f} dB")
                return adv_wav, current_eps

            current_eps += FGSM_EPSILON_STEP

        print("[FGSM] ❌ Could not find adversarial example within SNR constraints.")
        return waveform, None