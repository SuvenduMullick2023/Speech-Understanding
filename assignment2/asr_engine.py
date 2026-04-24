# ============================================================
# CELL 8: asr_engine.py  (Tasks 1.1, 1.2 – Constrained ASR)
# FIX: Added N-gram LM training on syllabus (Task 1.2 requirement).
#      Logit bias now incorporates N-gram scores, not just a flat map.
#      Kept full float32 fix and GlossaryLogitsProcessor.
# ============================================================

import os
import math
import torch
import numpy as np
from collections import defaultdict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    LogitsProcessor
)
from config import ASR_MODEL_NAME, TECHNICAL_GLOSSARY, SYLLABUS_TEXT_PATH, NGRAM_ORDER


# ------------------------------------------------------------------
# Task 1.2 Helper: Simple N-gram Language Model trained on syllabus
# ------------------------------------------------------------------
class NgramLanguageModel:
    """
    Character/word N-gram LM trained on Speech Course Syllabus text.
    Used to compute log-probability boosts for technical terms during
    Whisper's constrained beam search (logit biasing).
    """

    def __init__(self, order: int = 3):
        self.order   = order
        self.counts  = defaultdict(lambda: defaultdict(int))
        self.totals  = defaultdict(int)
        self.vocab   = set()

    def train(self, text: str):
        """Train on plain-text syllabus."""
        tokens = text.lower().split()
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.order):
            ctx   = tuple(tokens[i: i + self.order - 1])
            word  = tokens[i + self.order - 1]
            self.counts[ctx][word] += 1
            self.totals[ctx]       += 1
        print(f"[NgramLM] Trained {self.order}-gram LM on {len(tokens)} tokens, vocab={len(self.vocab)}")

    def log_prob(self, context: tuple, word: str) -> float:
        """Smoothed log-probability P(word | context)."""
        ctx   = tuple(context[-(self.order - 1):])
        count = self.counts[ctx].get(word, 0)
        total = self.totals[ctx]
        # Add-1 (Laplace) smoothing
        prob  = (count + 1) / (total + len(self.vocab) + 1)
        return math.log(prob)

    @classmethod
    def load_or_train(cls, text_path: str, order: int = 3) -> "NgramLanguageModel":
        lm = cls(order=order)
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                lm.train(f.read())
        else:
            # Fallback: tiny seed corpus of technical terms if syllabus file missing
            seed = " ".join(TECHNICAL_GLOSSARY.keys()) * 20
            print(f"[NgramLM] WARNING: {text_path} not found. Using seed corpus.")
            lm.train(seed)
        return lm


# ------------------------------------------------------------------
# Task 1.2: Logit Biasing using Glossary + N-gram LM scores
# ------------------------------------------------------------------
class GlossaryLogitsProcessor(LogitsProcessor):
    """
    Boosts token scores for technical glossary terms during Whisper decoding.
    Combines a static bias with N-gram LM log-probability for the current context.
    """

    def __init__(self, token_bias_map: dict, lm: NgramLanguageModel,
                 bias_strength: float = 5.0, lm_weight: float = 1.0):
        self.bias_map     = token_bias_map   # {term: [token_ids]}
        self.lm           = lm
        self.bias_strength = bias_strength
        self.lm_weight    = lm_weight
        self._recent_tokens: list = []       # rolling context window

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Update rolling context from last decoded token
        if input_ids.shape[1] > 0:
            last_token_id = input_ids[0, -1].item()
            self._recent_tokens.append(str(last_token_id))
            if len(self._recent_tokens) > self.lm.order:
                self._recent_tokens.pop(0)

        context = tuple(self._recent_tokens)

        for term, token_ids in self.bias_map.items():
            # LM bonus for first sub-token of each technical term
            lm_bonus = self.lm_weight * self.lm.log_prob(context, term.split()[0])
            for tid in token_ids:
                if tid < scores.shape[-1]:
                    scores[:, tid] += self.bias_strength + lm_bonus
        return scores


# ------------------------------------------------------------------
# Task 1.2: Constrained Whisper ASR
# ------------------------------------------------------------------
class ConstrainedWhisperASR:
    """
    Whisper-medium with:
      - GlossaryLogitsProcessor (static bias + N-gram LM boost)
      - Full float32 (no FP16 dtype mismatch on Colab T4)
    """

    def __init__(self):
        print(f"[ASR] Loading {ASR_MODEL_NAME} ...")
        self.processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
        self.model     = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL_NAME, low_cpu_mem_usage=True
        )
        self.tokenizer = self.processor.tokenizer
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Force full float32 (prevents dtype mismatch on Colab)
        self.model.float().to(self.device)
        print(f"[ASR] Model on {self.device}, dtype={next(self.model.parameters()).dtype}")

        # Train N-gram LM on syllabus
        self.lm = NgramLanguageModel.load_or_train(SYLLABUS_TEXT_PATH, NGRAM_ORDER)

        # Build glossary token bias map
        self.glossary_token_ids = self._prepare_glossary_bias()

    def _prepare_glossary_bias(self) -> dict:
        bias_map = {}
        for term in TECHNICAL_GLOSSARY.keys():
            ids = self.tokenizer.encode(term, add_special_tokens=False)
            if ids:
                bias_map[term] = ids
        return bias_map

    def transcribe_with_bias(self, audio_input: np.ndarray, language: str = "hi") -> str:
        """
        audio_input : numpy float32 array at 16 kHz (after denoising).
        language    : primary language hint to Whisper ('hi' for Hinglish).
        Returns     : transcribed string.
        """
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.float().to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        logits_processor = GlossaryLogitsProcessor(
            self.glossary_token_ids,
            lm=self.lm,
            bias_strength=5.0,
            lm_weight=1.0
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                logits_processor=[logits_processor],
                num_beams=5,
                max_length=448,
                no_repeat_ngram_size=3
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]