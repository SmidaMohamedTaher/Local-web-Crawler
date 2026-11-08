import re
import joblib
import json
from rank_bm25 import BM25Okapi

class My_BM25:

    def __init__(self, pages_data):

        self.pages_data = pages_data

        self.tokenized_docs = [
            self.tokenize(page["text"])
            for page in pages_data
        ]

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def tokenize(self, text):

        return re.findall(r"\w+", text.lower())

    def save(self, path="APIs/bm25_model.pkl"):

        joblib.dump(self, path)

    @classmethod
    def load(cls, path="APIs/bm25_model.pkl"):

        return joblib.load(path)

    def search(self, query):

        tokenized_query = self.tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        results = [
            {
                "title": page["title"],
                "url": page["url"],
                "score": float(score)
            }
            for score, page in zip(scores, self.pages_data)
            if score > 0
        ]

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results
