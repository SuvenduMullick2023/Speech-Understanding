import os

# --- Paths ---
DATA_DIR   = "/home/suvendu/speech_understanding/Speech-Understanding/assignment2/data"
OUTPUT_DIR = "/home/suvendu/speech_understanding/Speech-Understanding/assignment2/output"

REF_VOICE_PATH   = os.path.join(DATA_DIR, "student_voice_ref.wav")    # 60s student recording
SOURCE_AUDIO_PATH = os.path.join(DATA_DIR, "original_segment.wav")    # 10-min lecture clip
OUTPUT_LRL_PATH  = os.path.join(OUTPUT_DIR, "output_LRL_cloned.wav")  # Final synthesised output
DENOISED_PATH    = os.path.join(OUTPUT_DIR, "denoised_segment.wav")   # After Task 1.3

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ASR Model ---
# Using whisper-medium to balance accuracy and Colab RAM constraints.
# Upgrade to whisper-large-v3 on A100 runtime for best WER.
ASR_MODEL_NAME = "openai/whisper-medium"

# --- Target Low-Resource Language ---
# Bengali (bn) is chosen as the target LRL.
# NOTE: YourTTS supports Bengali via its multilingual VITS backend.
# If Bengali is unavailable in your TTS build, switch to "hi" as a
# fallback and note this limitation in your 1-page implementation note.
TARGET_LRL = "bn"

# --- Frame-level LID ---
FRAME_SHIFT_MS = 10       # 10 ms frames for LID
MIN_F1_LID     = 0.85     # Minimum required F1

# --- Evaluation Thresholds (Assignment §5) ---
MAX_WER_ENG      = 0.15   # WER < 15% for English segments
MAX_WER_HIN      = 0.25   # WER < 25% for Hindi segments
MAX_MCD          = 8.0    # Mel-Cepstral Distortion < 8.0
EER_THRESHOLD    = 0.10   # Equal Error Rate < 10%
LID_SWITCH_MS    = 200    # Timestamp precision for lang switches ≤ 200 ms
SNR_THRESHOLD_DB = 40.0   # Adversarial noise must keep SNR > 40 dB

# --- FGSM Search ---
FGSM_EPSILON_STEP = 0.001
MAX_FGSM_STEPS    = 50

# --- N-gram LM (Task 1.2) ---
# Path to the Speech Course Syllabus text file used to train the N-gram LM.
# Create this file manually by copying the course syllabus into plain text.
SYLLABUS_TEXT_PATH = os.path.join(DATA_DIR, "syllabus.txt")
NGRAM_ORDER = 3  # Trigram LM

# --- Technical Glossary: English term -> Bengali translation (Task 2.2) ---
# Expand this to ≥ 500 entries for full credit.
TECHNICAL_GLOSSARY = {
    "stochastic":           "স্টোকাস্টিক",
    "cepstrum":             "সেপস্ট্রাম",
    "fourier transform":    "ফুরিয়ার রূপান্তর",
    "spectrogram":          "স্পেকট্রোগ্রাম",
    "neural network":       "নিউরাল নেটওয়ার্ক",
    "hidden markov model":  "হিডেন মার্কভ মডেল",
    "dynamic time warping": "ডাইনামিক টাইম ওয়ার্পিং",
    "mel frequency":        "মেল ফ্রিকোয়েন্সি",
    "language model":       "ভাষা মডেল",
    "acoustic model":       "অ্যাকোস্টিক মডেল",
    "beam search":          "বিম সার্চ",
    "word error rate":      "ওয়ার্ড এরর রেট",
    "phoneme":              "ফোনিম",
    "prosody":              "প্রোসোডি",
    "fundamental frequency":"মূল কম্পাঙ্ক",
    # ... add remaining entries to reach 500
}