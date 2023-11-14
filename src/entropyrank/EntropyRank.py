import torch
import numpy as np
import spacy
import json
import os

from enum import Enum
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import defaultdict
from transformers import AutoTokenizer, GPTNeoForCausalLM
from keyphrase_vectorizers import KeyphraseCountVectorizer


class PartitionMethod(str, Enum):
    STOP_WORDS = "stop_words"
    NOUN_PHRASES = "noun_phrases"


class RankingMethod(str, Enum):
    SUM_ENTROPY = "sum_entropy"
    FIRST_WORD_ENTROPY = "first_word_entropy"


class EntropyRank:
    def __init__(
        self, model=None, tokenizer=None, stop_words=None, corpus=None, device=None
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.WORD_STATISTIC_PATH = "word_statistics.json"
        self.tokenizer = (
            AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            if not tokenizer
            else tokenizer
        )
        self.model = (
            GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            if not model
            else model
        )
        self.model.to(self.device)
        self.stop_words_files = [
            # "stopwords/FoxStoplist.txt",
            "stopwords/TerrierStopList.txt",
            "stopwords/SmartStoplist.txt",
        ]
        self.stop_words = stop_words if stop_words else self._generate_stop_words()
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 10000000
        self.key_phrase_vectorizer = KeyphraseCountVectorizer(
            spacy_pipeline=self.nlp, pos_pattern="<J.*>*<N.*>+"
        )

        self.entropy_word_statistics = (
            self._compute_entropy_word_statistics(corpus)
            if corpus
            else self._load_word_statistics()
        )
        # compute average between 100 to 500
        self.average_entropy = np.mean(
            [
                value
                for key, value in self.entropy_word_statistics.items()
                if key >= 100 and key <= 500
            ]
        )

    def extract_key_phrases_with_prefix(
        self,
        prefix: str,
        text: str,
        number_of_key_phrases=3,
        exclude_start_words_count=0,
        partition_method: PartitionMethod = PartitionMethod.NOUN_PHRASES,
        ranking_method: RankingMethod = RankingMethod.SUM_ENTROPY,
        normalize_by_word_statistics=False,
        remove_personal_names=False,
    ):
        concatenated_text = prefix + "\n" + text
        normalized_text = EntropyRank._normalize_text(concatenated_text)
        normalized_prefix = EntropyRank._normalize_text(prefix)
        tokenized_text = self._tokenize_text(normalized_text)
        tokenized_prefix = self._tokenize_text(normalized_prefix)
        words_to_token_indices_prefix = self._map_words_to_token_indices(
            tokenized_prefix
        )
        words_to_tokens_indices = self._map_words_to_token_indices(tokenized_text)
        entropy, relevant_tokens, _ = self._get_tokens_entropy(tokenized_text)

        decoded = self.tokenizer.batch_decode(relevant_tokens)
        original_words = EntropyRank._get_original_words(
            decoded, words_to_tokens_indices
        )

        return self._get_highest_entropy_candidates(
            normalized_text=normalized_text,
            original_words=original_words,
            entropy=entropy,
            number_of_results=number_of_key_phrases,
            remove_personal_names=remove_personal_names,
            words_to_tokens_indices=words_to_tokens_indices,
            exclude_start_words_count=len(words_to_token_indices_prefix),
            normalize_by_word_statistics=normalize_by_word_statistics,
            ranking_method=ranking_method,
            partition_method=partition_method,
        )

    def extract_key_phrases(
        self,
        text: str,
        number_of_key_phrases=3,
        exclude_start_words_count=0,
        partition_method: PartitionMethod = PartitionMethod.NOUN_PHRASES,
        ranking_method: RankingMethod = RankingMethod.SUM_ENTROPY,
        normalize_by_word_statistics=False,
        remove_personal_names=False,
        use_log_loss=False,
    ):
        normalized_text = EntropyRank._normalize_text(text)
        tokenized_text = self._tokenize_text(normalized_text)
        words_to_tokens_indices = self._map_words_to_token_indices(tokenized_text)
        entropy, relevant_tokens, _ = (
            self._get_tokens_entropy(tokenized_text)
            if not use_log_loss
            else self._get_tokens_log_loss(tokenized_text)
        )
        decoded = self.tokenizer.batch_decode(relevant_tokens)
        original_words = EntropyRank._get_original_words(
            decoded, words_to_tokens_indices
        )

        return self._get_highest_entropy_candidates(
            normalized_text=normalized_text,
            original_words=original_words,
            entropy=entropy,
            number_of_results=number_of_key_phrases,
            remove_personal_names=remove_personal_names,
            words_to_tokens_indices=words_to_tokens_indices,
            exclude_start_words_count=exclude_start_words_count,
            normalize_by_word_statistics=normalize_by_word_statistics,
            ranking_method=ranking_method,
            partition_method=partition_method,
        )

    def _get_highest_entropy_candidates(
        self,
        normalized_text: str,
        original_words: list[str],
        entropy: torch.Tensor,
        number_of_results: int,
        words_to_tokens_indices: list[tuple[int, int]],
        exclude_start_words_count=0,
        remove_personal_names=True,
        normalize_by_word_statistics=False,
        ranking_method: RankingMethod = RankingMethod.SUM_ENTROPY,
        partition_method: PartitionMethod = PartitionMethod.STOP_WORDS,
    ):
        entropy = EntropyRank._get_entropy_of_original_words(
            original_words, entropy, words_to_tokens_indices
        )

        if exclude_start_words_count > 0:
            original_words = original_words[exclude_start_words_count:]
            entropy = entropy[exclude_start_words_count:]

        # Remove personal names
        if remove_personal_names:
            person_tokens_indicators = self._get_person_words_map(normalized_text)
            original_words, entropy = self._remove_personal_names(
                original_words, entropy, person_tokens_indicators
            )

        if normalize_by_word_statistics:
            for index, value in enumerate(entropy):
                # we take only the first 100 words due to high variance in the entropy
                entropy[index] = (
                    value - self.entropy_word_statistics[index]
                    if index < 100
                    else value - self.average_entropy
                )

        partitioned_words, partitioned_entropy = self._partition(
            method=partition_method,
            original_words=original_words,
            entropy=entropy,
            normalized_text=normalized_text,
        )

        return self._get_top_k_by_partition(
            number_of_results=number_of_results,
            ranking_method=ranking_method,
            partitioned_words=partitioned_words,
            partitioned_entropy=partitioned_entropy,
        )

    @staticmethod
    def _get_original_words(
        decoded: list[str], words_to_tokens_indices: list[tuple[int, int]]
    ) -> list[str]:
        original_words = [
            "".join(decoded[start:end]).strip()
            for start, end in words_to_tokens_indices
        ]
        return original_words

    def _partition(
        self,
        method: PartitionMethod,
        original_words: list[str],
        entropy: list[float],
        normalized_text: str,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if method == PartitionMethod.STOP_WORDS:
            return self._partition_by_stop_words(original_words, entropy)
        else:
            return self._partition_by_noun_phrases(
                original_words, entropy, normalized_text
            )

    def _get_top_k_by_partition(
        self,
        number_of_results: int,
        partitioned_words: list[list[str]],
        partitioned_entropy: list[list[float]],
        ranking_method: RankingMethod = RankingMethod.SUM_ENTROPY,
    ):
        ADDITIONAL_PARTITIONS_FOR_DEDUP = 10
        if ranking_method == RankingMethod.SUM_ENTROPY:
            entropy_of_partitions = [
                sum(partitioned_entropy[i]) for i in range(len(partitioned_entropy))
            ]
        # first word entropy
        else:
            entropy_of_partitions = [
                partitioned_entropy[i][0] for i in range(len(partitioned_entropy))
            ]
        # Get number_of_results partitions with the highest average entropy
        top_k_partitions = EntropyRank._get_top_k_indices(
            entropy_of_partitions, number_of_results + ADDITIONAL_PARTITIONS_FOR_DEDUP
        )

        top_k_partitions_words = [partitioned_words[i] for i in top_k_partitions]
        top_k_partitions_entropy = [entropy_of_partitions[i] for i in top_k_partitions]

        # dedup partitions
        (
            deduped_partitions_words,
            deduped_partitions_entropy,
        ) = EntropyRank._dedup_partitions(
            top_k_partitions_words, top_k_partitions_entropy, number_of_results
        )

        # concat the deduped partitions words into a list of phrases
        phrases = [" ".join(partition) for partition in deduped_partitions_words]
        # if - is present remove white space before and after it
        phrases = [phrase.replace(" - ", "-") for phrase in phrases]

        return list(zip(phrases, deduped_partitions_entropy))

    @staticmethod
    def _dedup_partitions(
        top_k_partitions_words, top_k_partitions_entropy, number_of_results: int
    ):
        deduped_partitions_words = []
        lower_case_partitions_words = [
            [word.lower() for word in partition] for partition in top_k_partitions_words
        ]
        lower_case_deduped_partitions_words = []
        deduped_partitions_entropy = []
        for i in range(len(top_k_partitions_words)):
            if (
                lower_case_partitions_words[i]
                not in lower_case_deduped_partitions_words
            ):
                deduped_partitions_words.append(top_k_partitions_words[i])
                deduped_partitions_entropy.append(top_k_partitions_entropy[i])
                lower_case_deduped_partitions_words.append(
                    lower_case_partitions_words[i]
                )

        # take only the top k partitions
        top_k_partitions_words = deduped_partitions_words[:number_of_results]
        top_k_partitions_entropy = deduped_partitions_entropy[:number_of_results]

        return top_k_partitions_words, top_k_partitions_entropy

    def _remove_personal_names(
        self,
        original_words: list[str],
        entropy: list[float],
        person_tokens_indicators: dict[str, list[int]],
    ) -> tuple[list[str], list[float]]:
        new_original_words = []
        new_entropy = []
        # Convert entropy from list of tensors to list of floats
        entropy = [float(entropy[i]) for i in range(len(entropy))]
        for i in range(len(original_words)):
            if not person_tokens_indicators[original_words[i]]:
                new_original_words.append(original_words[i])
                new_entropy.append(entropy[i])

        return new_original_words, new_entropy

    # TODO: since spacy tokenization doesn't match huggingface tokenization, we're using a map instead of a list of indices
    # This can create problems if the same name appears multiple times in the text and has multiple meanings
    def _get_person_words_map(self, text: str) -> dict[str, list[int]]:
        doc = self.nlp(text)
        # default dict with False values
        person_tokens_indicators = defaultdict(lambda: False)

        for token in doc:
            # if token.text exists in default dict and is False don't change it
            if (
                token.text in person_tokens_indicators
                and not person_tokens_indicators[token.text]
            ):
                continue
            person_tokens_indicators[token.text] = token.ent_type_ == "PERSON"

        return person_tokens_indicators

    @torch.no_grad()
    def _get_tokens_entropy(
        self, tokenized_text: str
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        # convert tokenized text to gpu tensors
        tokenized_text = {k: v.to(self.device) for k, v in tokenized_text.items()}
        # Get probabilities from logits
        logits = self.model(**tokenized_text).logits

        # Remove bos token from tokenized email
        relevant_tokens = tokenized_text["input_ids"][0][1:]

        entropy = EntropyRank._calculate_entropy_from_logits(logits)

        return entropy, relevant_tokens, logits

    @torch.no_grad()
    def _get_tokens_log_loss(
        self, tokenized_text: str
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        # convert tokenized text to gpu tensors
        tokenized_text = {k: v.to(self.device) for k, v in tokenized_text.items()}
        # Get probabilities from logits
        logits = self.model(**tokenized_text).logits

        # Remove bos token from tokenized email
        relevant_tokens = tokenized_text["input_ids"][0][1:]

        log_loss = EntropyRank._calculate_log_loss_from_logits(logits, relevant_tokens)

        return log_loss, relevant_tokens, logits

    @staticmethod
    def _load_stop_words_from_file(path: str) -> set[str]:
        stop_words = set()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_dir, path)
        with open(path, "r") as f:
            for line in f:
                stop_words.add(line.strip())
        return stop_words

    def _generate_stop_words_from_files(self) -> set[str]:
        stop_words_from_files = set()
        for file_path in self.stop_words_files:
            stop_words_from_files.update(
                EntropyRank._load_stop_words_from_file(file_path)
            )

        return stop_words_from_files

    def _generate_stop_words(self) -> set[str]:
        stop_words_from_files = self._generate_stop_words_from_files()
        determiners = ["the", "a", "an", "this", "that", "these", "those"]
        conjunctions = ["and", "or", "but", "because", "so", "yet", "for"]
        pronouns = [
            "he",
            "she",
            "it",
            "they",
            "we",
            "you",
            "I",
            "me",
            "him",
            "her",
            "them",
            "us",
            "you",
        ]
        modals = ["can", "could", "may", "might", "must", "will", "would"]
        # remove subject pronouns
        subject_pronouns = ["i", "you", "he", "she", "it", "we", "they"]

        # Create a set out of the following lists
        stop_words = set(STOP_WORDS)
        stop_words.update(stop_words_from_files)

        # add variants without ' to stop words
        words_to_add = []
        for word in stop_words:
            if "'" in word:
                words_to_add.append(word.replace("'", ""))

        stop_words.update(words_to_add)

        stop_words.update(determiners)
        stop_words.update(conjunctions)
        stop_words.update(pronouns)
        stop_words.update(punctuation)
        stop_words.update(modals)
        stop_words.update(subject_pronouns)

        stop_words.remove("-")
        stop_words.remove("/")
        return stop_words

    def _tokenize_text(self, text: str):
        tokenized_text = self.tokenizer(
            self.tokenizer.bos_token + text,
            return_tensors="pt",
            truncation=True,
        )
        return tokenized_text

    @staticmethod
    def _check_if_multiple_punctuation(token: str) -> bool:
        return len(token) > 1 and token[0] in punctuation and token[-1] in punctuation

    def _is_stop_word(self, token: str) -> bool:
        token = token.lower()
        return (
            token in self.stop_words
            or len(token) < 2
            or EntropyRank._check_if_multiple_punctuation(token)
            or token.isnumeric()
        )

    def _partition_by_noun_phrases(
        self, original_words: list[str], entropy: list[float], normalized_text: str
    ) -> tuple[list[list[str]], list[list[float]]]:
        noun_phrases_indices = self._get_noun_phrases_indices(
            normalized_text, original_words
        )
        partitioned_tokens_strings = []
        partitioned_entropies = []
        for start_index, end_index in noun_phrases_indices:
            partitioned_tokens_strings.append(original_words[start_index:end_index])
            partitioned_entropies.append(entropy[start_index:end_index])

        # Remove empty partitions
        partitioned_tokens_strings = [
            partition for partition in partitioned_tokens_strings if partition
        ]
        partitioned_entropies = [
            partition for partition in partitioned_entropies if partition
        ]

        return partitioned_tokens_strings, partitioned_entropies

    def _partition_by_stop_words(
        self, original_words: list[str], entropy: list[float]
    ) -> tuple[list[list[str]], list[list[float]]]:
        # Concat word to the previous word if it's "'s"
        (
            original_words,
            entropy,
        ) = EntropyRank._concat_apostrophes_to_previous_word(original_words, entropy)
        # create a list of lists of tokens and entropies
        # each list of tokens and entropies is a partition which is separated by stop words
        partitioned_tokens_strings = []
        partitioned_entropies = []
        current_partition_index = 0
        for i in range(len(original_words)):
            if self._is_stop_word(original_words[i]):
                partitioned_tokens_strings.append(
                    original_words[current_partition_index:i]
                )
                partitioned_entropies.append(entropy[current_partition_index:i])
                current_partition_index = i + 1
        # add the last partition
        partitioned_tokens_strings.append(original_words[current_partition_index:])
        partitioned_entropies.append(entropy[current_partition_index:])

        # Remove empty partitions
        partitioned_tokens_strings = [
            partition for partition in partitioned_tokens_strings if partition
        ]
        partitioned_entropies = [
            partition for partition in partitioned_entropies if partition
        ]

        return partitioned_tokens_strings, partitioned_entropies

    @staticmethod
    def _concat_apostrophes_to_previous_word(
        token_strings: list[str], entropies: list[float]
    ) -> tuple[list[str], list[float]]:
        new_token_strings = []
        new_entropies = []

        if len(token_strings) != len(entropies):
            return token_strings, entropies
        # Concatenate apostrophes to previous word
        for i in range(len(token_strings)):
            if token_strings[i] == "'s" or token_strings[i] == "'t":
                new_token_strings[-1] += token_strings[i]
                new_entropies[-1] += entropies[i]
            else:
                new_token_strings.append(token_strings[i])
                new_entropies.append(entropies[i])

        return new_token_strings, new_entropies

    @staticmethod
    def _get_top_k_indices(input_list: list[float], k: int) -> list[int]:
        # Convert to numpy array
        input_list = np.array(input_list)
        # Get the indices of the top k values
        top_k_indices = input_list.argsort()[-k:][::-1]
        # return as list
        return top_k_indices.tolist()

    @staticmethod
    def _get_entropy_of_original_words(
        original_words: list[str],
        entropy: torch.Tensor,
        words_to_token_indices: list[tuple[int, int]],
    ) -> list[float]:
        mapped_entropy = []
        if len(original_words) != len(words_to_token_indices):
            print("Error in mapping entropy to original words")
        for start, end in words_to_token_indices:
            mapped_entropy.append(entropy[start:end].sum().item())

        return mapped_entropy

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\t", " ")
        return text

    @staticmethod
    def _calculate_log_loss_from_logits(logits, relevant_tokens):
        probs = logits[0].softmax(dim=-1)

        # Get the actual probs of each token
        actual_probs = probs[range(len(probs) - 1), relevant_tokens]

        # Get cross entropy of each token
        log_loss = -actual_probs.log2()
        return log_loss

    @staticmethod
    def _calculate_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # make these calculations on the cpu
        probs = logits[0].softmax(dim=-1)
        # We don't need prediction for the last token
        relevant_probs = probs[range(len(probs) - 1), :]
        entropy = -torch.sum(relevant_probs * torch.log2(relevant_probs), dim=-1)

        # check if there is nan here - this happens when the probability is really close to 0
        if torch.isnan(entropy).any():
            # replace any nan with 0
            entropy[torch.isnan(entropy)] = 0

        return entropy

    def _get_noun_phrases_indices(
        self, text: str, original_words: list[str]
    ) -> list[tuple[int, int]]:
        vectorizer_phrases = self._match_with_vectorizer(text)
        vectorized_indices = self._find_matching_noun_phrase_indices(
            vectorizer_phrases, original_words
        )
        return vectorized_indices

    def _find_matching_noun_phrase_indices(
        self, noun_phrases: np.array(str), original_words: list[str]
    ) -> list[tuple[int, int]]:
        # convert original words to lower case
        original_words = [word.lower() for word in original_words]
        # delimiter noun phrases by space
        noun_phrases = [phrase.split(" ") for phrase in noun_phrases]
        # find indices of noun phrases
        indices = []
        # look for all the substrings which match a noun phrase
        for i in range(len(original_words) + 1):
            for j in range(i, len(original_words) + 1):
                if original_words[i:j] in noun_phrases:
                    indices.append((i, j))

        return indices

    def _match_with_vectorizer(self, text: str) -> np.array(str):
        self.key_phrase_vectorizer.fit_transform([text])
        return self.key_phrase_vectorizer.get_feature_names_out()

    def _load_word_statistics(self) -> dict[int, float]:
        # get current directory
        current_directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_directory, self.WORD_STATISTIC_PATH)
        with open(path, "r") as file:
            statistics = json.load(file)

        # convert keys to int
        statistics = {int(key): value for key, value in statistics.items()}
        return statistics

    def _compute_entropy_word_statistics(self, corpus: list[str]) -> dict[int, float]:
        # create a default dict with empty list as default value
        statistics = dict()
        histogram = dict()
        for text in corpus:
            normalized_text = EntropyRank._normalize_text(text)
            tokenized_text = self._tokenize_text(normalized_text)
            words_to_tokens_indices = self._map_words_to_token_indices(tokenized_text)
            entropy, _, _ = self._get_tokens_entropy(tokenized_text)
            original_words = self._get_original_words(normalized_text)
            entropy = EntropyRank._get_entropy_of_original_words(
                original_words, entropy, words_to_tokens_indices
            )
            # Concat word to the previous word if it's "'s"
            (
                original_words,
                entropy,
            ) = EntropyRank._concat_apostrophes_to_previous_word(
                original_words, entropy
            )
            for index, value in enumerate(entropy):
                statistics[index] = statistics.get(index, 0) + value
                histogram[index] = histogram.get(index, 0) + 1

        # compute average entropy
        for index, value in statistics.items():
            statistics[index] = value / histogram[index]

        return statistics

    @staticmethod
    def _map_words_to_token_indices(encoded) -> list[tuple[int, int]]:
        desired_output = []
        for word_id in encoded.word_ids():
            if word_id is not None:
                start, end = encoded.word_to_tokens(word_id)
                # we subtract 1 from start because the first token is BOS token,
                tokens = (start - 1, end - 1)
                if len(desired_output) == 0 or desired_output[-1] != tokens:
                    desired_output.append(tokens)

        # remove first token index because it's BOS token
        return desired_output[1:]
