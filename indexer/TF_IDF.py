import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

class My_TFIDF :
    def __init__(self, pages_data):
        self.pages_data = pages_data
        self.documents = [page["text"] for page in pages_data]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def get_tfidf_matrix(self):
        return self.tfidf_matrix
    
    def get_tfidf_df(self):
        return pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    def query_vector(self, query):
        return self.vectorizer.transform([query])

    def search(self, query, threshold=0.0):
        
        query_vec = self.query_vector(query)
        scores = cosine_similarity(
            query_vec,
            self.tfidf_matrix
        ).flatten()

        ranked_indices = scores.argsort()[::-1]

        results = []

        for idx in ranked_indices:
            if scores[idx] >= threshold :
                results.append({
                    "title": self.pages_data[idx]["title"],
                    "url": self.pages_data[idx]["url"],
                    "score": scores[idx]
                })

        return results

    def save(self, path="APIs/tfidf_model.pkl"):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path="APIs/tfidf_model.pkl"):
        return joblib.load(path)

# with open("APIs/crawled_pages.json", "r", encoding="utf-8") as file:
#      pages_data = json.load(file)
     
# tfidf = My_TFIDF(pages_data)
# tfidf.save()