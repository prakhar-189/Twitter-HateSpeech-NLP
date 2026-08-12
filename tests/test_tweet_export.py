import unittest

from tweet_export import MAX_EXPORT_BYTES, load_tweet_texts


class TweetExportTest(unittest.TestCase):
    def test_loads_json_array_text_fields(self):
        payload = b'[{"text":"first tweet"},{"full_text":"second tweet"}]'

        self.assertEqual(load_tweet_texts(payload, "tweets.json"), ["first tweet", "second tweet"])

    def test_loads_wrapped_export_records(self):
        payload = b'{"data":[{"tweetText":"wrapped tweet"}]}'

        self.assertEqual(load_tweet_texts(payload, "export.json"), ["wrapped tweet"])

    def test_loads_jsonl_records(self):
        payload = b'{"content":"one"}\n{"body":"two"}\n'

        self.assertEqual(load_tweet_texts(payload, "tweets.jsonl"), ["one", "two"])

    def test_loads_csv_records(self):
        payload = b"id,text\n1,hello from csv\n2,second row\n"

        self.assertEqual(load_tweet_texts(payload, "tweets.csv"), ["hello from csv", "second row"])

    def test_rejects_exports_without_text(self):
        with self.assertRaisesRegex(ValueError, "No tweet text fields"):
            load_tweet_texts(b'[{"id":"1"}]', "tweets.json")

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            load_tweet_texts(b'{"text":', "tweets.json")

    def test_rejects_non_utf8_exports(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_tweet_texts(b"\xff", "tweets.json")

    def test_rejects_oversized_exports(self):
        with self.assertRaisesRegex(ValueError, "5 MB"):
            load_tweet_texts(b"x" * (MAX_EXPORT_BYTES + 1), "tweets.json")


if __name__ == "__main__":
    unittest.main()
