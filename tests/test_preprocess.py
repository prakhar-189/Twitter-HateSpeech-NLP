from src.preprocess import clean_tweets


def test_removes_user_handles():
    cleaned, _ = clean_tweets(["@user hello world"])
    assert "@user" not in cleaned[0]
    assert "user" not in cleaned[0].split()


def test_removes_urls():
    cleaned, _ = clean_tweets(["check this out http://t.co/abc123 www.example.com"])
    assert "http" not in cleaned[0]
    assert "www.example.com" not in cleaned[0]


def test_lowercases_text():
    cleaned, _ = clean_tweets(["HATE Speech Is BAD"])
    assert cleaned[0] == cleaned[0].lower()


def test_removes_stopwords():
    # "is", "a", "the" are stopwords and should not survive
    cleaned, _ = clean_tweets(["this is a the test"])
    for stopword in ("is", "a", "the"):
        assert stopword not in cleaned[0].split()


def test_removes_redundant_terms_amp_and_rt():
    cleaned, _ = clean_tweets(["rt this amp that terrible"])
    assert "rt" not in cleaned[0].split()
    assert "amp" not in cleaned[0].split()


def test_strips_hashtag_symbol_but_keeps_text():
    cleaned, _ = clean_tweets(["so #sad about this"])
    assert "sad" in cleaned[0].split()
    assert "#sad" not in cleaned[0].split()


def test_removes_single_character_tokens():
    cleaned, _ = clean_tweets(["a b hello c world"])
    assert "hello" in cleaned[0].split()
    assert "world" in cleaned[0].split()
    for tok in ("a", "b", "c"):
        assert tok not in cleaned[0].split()


def test_all_tokens_aggregated_across_tweets():
    _, all_tokens = clean_tweets(["hello world", "hello again"])
    assert all_tokens.count("hello") == 2


def test_empty_string_input_produces_empty_output():
    cleaned, all_tokens = clean_tweets([""])
    assert cleaned == [""]
    assert all_tokens == []


def test_non_string_input_is_coerced_to_string():
    # pandas can hand this function NaN/float values from a malformed CSV row;
    # str(tweet) in clean_tweets should not raise.
    cleaned, _ = clean_tweets([123, None])
    assert len(cleaned) == 2
