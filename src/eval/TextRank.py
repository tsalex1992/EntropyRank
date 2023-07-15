import spacy
import pytextrank
from spacy.lang.en.stop_words import STOP_WORDS


class TextRank:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

    # examine the top-ranked phrases in the document
    def extract_key_phrases(self, text):
        doc = self.nlp(text)
        phrases = []
        for phrase in doc._.phrases:
            phrases.append(self._remove_stop_words_from_start(phrase.text))

        return self._remove_duplicate_phrases(phrases)

    def _remove_stop_words_from_start(self, text):
        words = text.split()
        for index, word in enumerate(words):
            if word.lower() in STOP_WORDS:
                continue
            else:
                return " ".join(words[index:])

        return " ".join(words)

    def _remove_duplicate_phrases(self, phrases):
        unique_phrases = []
        for phrase in phrases:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
        return unique_phrases
