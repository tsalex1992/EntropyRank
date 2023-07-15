from string import punctuation


def prepare_ds_for_eval(dataset) -> tuple[list[str], list[list[str]]]:
    # get the names of the splits
    marking_tokens = _find_marking_tokens_in_ds(dataset)
    docs = _get_all_docs_from_ds_splits(dataset)
    # remove the marking tokens from the docs
    docs = [_remove_marking_tokens_from_doc(doc, marking_tokens) for doc in docs]
    # join docs by space
    docs = [" ".join(doc) for doc in docs]
    # remove spaces before punctuation
    for p in punctuation:
        docs = [doc.replace(f" {p}", p) for doc in docs]

    # get all the phrases
    phrases = _get_all_phrases_from_ds_splits(dataset)
    # assert that the number of phrases is the same as the number of docs
    assert len(phrases) == len(docs)
    # filter out empty phrases
    return _filter_empty_phrases(docs, phrases)


def _filter_empty_phrases(docs, phrases) -> tuple[list[str], list[list[str]]]:
    # filter out empty phrases
    docs, phrases = zip(*[(doc, phrase) for doc, phrase in zip(docs, phrases) if len(phrase) > 0])
    return list(docs), list(phrases)


def _get_all_phrases_from_ds_splits(dataset) -> list[list[str]]:
    phrases = []
    for split_name in dataset.column_names.keys():
        split = dataset[split_name]
        phrases.extend(split["extractive_keyphrases"])

    return phrases


def _get_all_docs_from_ds_splits(dataset) -> list[list[str]]:
    docs = []
    for split_name in dataset.column_names.keys():
        split = dataset[split_name]
        docs.extend(split["document"])

    return docs


def _remove_marking_tokens_from_doc(doc, marking_tokens) -> list[str]:
    return [token for token in doc if token not in marking_tokens]


def _find_marking_tokens_in_ds(dataset) -> set:
    marking_tokens = set()
    for split_name in dataset.column_names.keys():
        split = dataset[split_name]
        marking_tokens = marking_tokens.union(_find_marking_tokens(split["document"]))
    return marking_tokens


def _find_marking_tokens(docs) -> set:
    tokens = set()
    for doc in docs:
        for token in doc:
            # check if token is for example "-word-"
            if token.startswith("-") and token.endswith("-") and len(token) > 1:
                tokens.add(token)
    return tokens
