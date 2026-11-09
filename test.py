
import json
import joblib
from search.search import Search
from indexer.BM25 import My_BM25
from indexer.TF_IDF import My_TFIDF


# Load TF-IDF model
tfidf = joblib.load("APIs/tfidf_model.pkl")

# Load BM25 model
bm25 = joblib.load("APIs/bm25_model.pkl")

# Load PageRank scores
with open("APIs/pagerank_scores.json", "r", encoding="utf-8") as file:
    pagerank = json.load(file)


# Create Search object
search_engine = Search(tfidf, bm25, pagerank)


# Display results
def display_results(results):

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 40)

        print("Title:", result["title"])
        print("URL:", result["url"])
        print("Score:", result["score"])


# Main program
def main():

    query = input("Enter your search query: ")

    print("\nChoose search method:")
    print("1. TF-IDF")
    print("2. BM25")
    print("3. TF-IDF + PageRank")
    print("4. BM25 + PageRank")

    choice = input("\nYour choice: ")

    if choice == "1":
        results = search_engine.search_tfidf(query)

    elif choice == "2":
        results = search_engine.search_bm25(query)

    elif choice == "3":
        results = search_engine.search_tfidf_pagerank(query)

    elif choice == "4":
        results = search_engine.search_bm25_pagerank(query)

    else:
        print("Invalid choice.")
        return

    display_results(results)


if __name__ == "__main__":
    main()


