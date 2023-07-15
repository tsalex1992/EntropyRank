# EntropyBasedKeyPhraseExtraction
This is the official implementation of the EntropyRank key phrase extractor from https://openreview.net/forum?id=WCTtOfIhsJ. Please cite the paper and star this repo if you find EntropyRank useful! Thanks!

```
@inproceedings{
tsvetkov2023entropyrank,
title={EntropyRank: Unsupervised Keyphrase Extraction via Side-Information Optimization for Language Model-based Text Compression},
author={Alexander Tsvetkov and Alon Kipnis},
booktitle={ICML 2023 Workshop Neural Compression: From Information Theory to Applications},
year={2023},
url={https://openreview.net/forum?id=WCTtOfIhsJ}
}
```
## Installation

To install directly:

```
pip install entropyrank
```

To install from repository, from src/entropyrank run:

```
pip install -r requirements.txt
```

You also need to download the 'en_core_web_sm' model for spaCy, which can be done by running:

```
spacy download en_core_web_sm
```

## Usage

To use the package, import `EntropyRank` from the module and create an instance of it:

```python
from entropyrank import EntropyRank

extractor = EntropyRank()
```

Then, you can extract key phrases from a given text using the `extract_key_phrases` method:

```python
phrases = extractor.extract_key_phrases(
    text=text,
    number_of_key_phrases=3,
)
```

The parameters of the `extract_key_phrases` method are:

- `text`: the input text to extract key phrases from.
- `number_of_key_phrases`: the number of key phrases to extract.
- `exclude_start_words_count`: the number of words to exclude from the start of each key phrase when calculating its entropy.
- `partition_method`: can be STOP_WORDS or NOUN_PHRASES, decides how to partition the candidates.
- `ranking_method`: can be FIRST_WORD_ENTROPY or SUM_ENTROPY, whether to use the sum of entropy of the phrase or just the entropy of the first word
- `normalize_by_word_statistics`: a boolean indicating whether we want to normalize the entropy values by entropy statistics of word position.
- `remove_personal_names`: a boolean indicating whether to remove personal names from the evaluations or not.

## Evaluation Demo

You can run the evaluation_demo notebook included in this repository under src/eval to get the benchmark results on common key phrase extraction tasks reported in the paper.
Make sure to run pip install -r evaluation-requirements.txt beforehand
