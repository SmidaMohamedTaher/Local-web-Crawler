import json
import os
import threading

import joblib

from crawler.crawler import crawl
from indexer.BM25 import My_BM25
from indexer.PageRanking import WebGraph
from indexer.TF_IDF import My_TFIDF


CRAWLED_PATH = "APIs/crawled_pages.json"
BM25_PATH = "APIs/bm25_model.pkl"
TFIDF_PATH = "APIs/tfidf_model.pkl"
PAGERANK_PATH = "APIs/pagerank_scores.json"
START_PAGE = "pages\\python.html"
INTERVAL = 10

_stop_event = threading.Event()
_data_lock = threading.Lock()


def _atomic_json(path, data):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(temporary_path, path)


def _atomic_pickle(path, data):
    temporary_path = path + ".tmp"
    joblib.dump(data, temporary_path)
    os.replace(temporary_path, path)


def _read_crawled_pages():
    with _data_lock:
        with open(CRAWLED_PATH, "r", encoding="utf-8") as file:
            return json.load(file)


def _run_crawler():
    while not _stop_event.is_set():
        try:
            pages_data = crawl(START_PAGE)
            with _data_lock:
                _atomic_json(CRAWLED_PATH, pages_data)
        except Exception as error:
            print(f"[Crawler] Error: {error}")
        _stop_event.wait(INTERVAL)


def _run_bm25():
    while not _stop_event.is_set():
        try:
            model = My_BM25(_read_crawled_pages())
            _atomic_pickle(BM25_PATH, model)
        except Exception as error:
            print(f"[BM25] Error: {error}")
        _stop_event.wait(INTERVAL)


def _run_tfidf():
    while not _stop_event.is_set():
        try:
            model = My_TFIDF(_read_crawled_pages())
            _atomic_pickle(TFIDF_PATH, model)
        except Exception as error:
            print(f"[TF-IDF] Error: {error}")
        _stop_event.wait(INTERVAL)


def _run_pagerank():
    while not _stop_event.is_set():
        try:
            graph = WebGraph(_read_crawled_pages())
            with _data_lock:
                scores = graph.calculate_pagerank()
                _atomic_json(PAGERANK_PATH, scores)
        except Exception as error:
            print(f"[PageRank] Error: {error}")
        _stop_event.wait(INTERVAL)


def start_workers():
    workers = []
    for target in (_run_crawler, _run_bm25, _run_tfidf, _run_pagerank):
        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        workers.append(worker)
    return workers


def stop_workers(workers):
    _stop_event.set()
    for worker in workers:
        worker.join(timeout=INTERVAL + 2)


def main():
    workers = start_workers()
    print(
        "Started crawler, BM25, TF-IDF, and PageRank workers. "
        f"They rerun every {INTERVAL} seconds. Press Ctrl+C to stop."
    )

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping workers...")
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    main()
