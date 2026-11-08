
class Search:

    def __init__(self, tfidf, bm25, pagerank):
        self.tfidf = tfidf
        self.bm25 = bm25
        self.pagerank = pagerank

    def search_tfidf(self, query):
        results = self.tfidf.search(query)
        return results
    
    def search_bm25(self,query):
        results = self.bm25.search(query)
        return results
    
    def normalize_scores(self, results):

        scores = [result["score"] for result in results]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    def search_tfidf_pagerank(self, query, alpha=0.8):

        results = self.tfidf.search(query)

        if not results:
            return []

        normalized_scores = self.normalize_scores(results)

        for result, score in zip(results, normalized_scores):

            pagerank_score = self.pagerank.get(result["url"], 0)

            result["score"] = (
                alpha * score +
                (1 - alpha) * pagerank_score
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def search_bm25_pagerank(self, query, alpha=0.8):

        results = self.bm25.search(query)

        if not results:
            return []

        normalized_scores = self.normalize_scores(results)

        for result, score in zip(results, normalized_scores):

            pagerank_score = self.pagerank.get(result["url"], 0)

            result["score"] = (
                alpha * score +
                (1 - alpha) * pagerank_score
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        return results