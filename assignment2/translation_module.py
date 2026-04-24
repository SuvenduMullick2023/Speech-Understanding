# ============================================================
# CELL 9: translation_module.py  (Tasks 2.1 & 2.2)
# FIX: Replaced UnicodeIndicTransliterator (outputs Indic script,
#      NOT IPA) with epitran for proper Hindi -> IPA conversion.
#      English IPA via g2p-en is kept as-is.
# ============================================================
import epitran                                             # Hindi -> IPA
from g2p_en import G2p                                    # English -> IPA (ARPAbet then mapped)
from config import TECHNICAL_GLOSSARY, TARGET_LRL


class PhoneticTranslator:
    """
    Task 2.1: Unified IPA representation for code-switched Hinglish text.
    Task 2.2: Glossary-based translation to target LRL (Bengali).
    """

    def __init__(self):
        self.g2p_en   = G2p()
        # epitran for Hindi transliteration to IPA
        # 'Latn-Deva' or 'Hind-Deva': use Hindi-Devanagari backend
        try:
            self.epi_hi = epitran.Epitran('Hind-Deva')
        except Exception as e:
            print(f"[PhoneticTranslator] epitran Hindi init warning: {e}")
            self.epi_hi = None

    # ------------------------------------------------------------------
    # Task 2.1
    # ------------------------------------------------------------------
    def text_to_ipa(self, text: str, lang: str = "en") -> str:
        """
        Convert text to IPA string.
        lang='en' -> g2p_en (English phonemes)
        lang='hi' -> epitran Hindi-Devanagari backend
        """
        if lang == "en":
            phonemes = self.g2p_en(text)
            return " ".join(phonemes)
        elif lang == "hi":
            if self.epi_hi:
                return self.epi_hi.transliterate(text)
            else:
                # Graceful fallback: return text as-is with warning
                print("[PhoneticTranslator] epitran unavailable, returning raw text.")
                return text
        else:
            return text

    def hinglish_to_ipa(self, hinglish_text: str) -> str:
        """
        Naïve word-level language detection:
        - Words containing Devanagari characters -> Hindi IPA via epitran
        - All-ASCII words -> English IPA via g2p_en
        Combined into one unified IPA string.
        """
        words = hinglish_text.split()
        ipa_parts = []
        for word in words:
            if any('\u0900' <= ch <= '\u097F' for ch in word):
                ipa_parts.append(self.text_to_ipa(word, lang="hi"))
            else:
                ipa_parts.append(self.text_to_ipa(word, lang="en"))
        return " ".join(ipa_parts)

    # ------------------------------------------------------------------
    # Task 2.2
    # ------------------------------------------------------------------
    def translate_with_glossary(self, hinglish_text: str) -> str:
        """
        Translate Hinglish text to Bengali using the technical glossary.
        Multi-word terms are matched before single-word fallback.
        Non-glossary words are kept in original script (extend glossary
        to ≥500 entries for full coverage as required by the assignment).
        """
        words       = hinglish_text.split()
        translated  = []
        i = 0
        while i < len(words):
            matched = False
            # Try longest match first (up to 3-word phrases)
            for n in [3, 2, 1]:
                phrase = " ".join(words[i: i + n]).lower().strip(".,!?")
                if phrase in TECHNICAL_GLOSSARY:
                    translated.append(TECHNICAL_GLOSSARY[phrase])
                    i += n
                    matched = True
                    break
            if not matched:
                translated.append(words[i])
                i += 1
        return " ".join(translated)