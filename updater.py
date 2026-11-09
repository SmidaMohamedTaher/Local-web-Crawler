import json
import os
import time
import threading
import joblib

from crawler.crawler import crawl
from indexer.BM25 import My_BM25
from indexer.TF_IDF import My_TFIDF
from indexer.PageRanking import WebGraph

CRAWLED_PATH   = "APIs/crawled_pages.json"
BM25_PATH      = "APIs/bm25_model.pkl"
TFIDF_PATH     = "APIs/tfidf_model.pkl"
PAGERANK_PATH  = "APIs/pagerank_scores.json"
START_PAGE     = "pages\\python.html"
INTERVAL       = 10

_stop_event = threading.Event()
_lock       = threading.Lock()


def _atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def _atomic_pkl(path, obj):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _run_crawler():
    while not _stop_event.is_set():
        try:
            pages_data = crawl(START_PAGE)
            with _lock:
                _atomic_json(CRAWLED_PATH, pages_data)
        except Exception as e:
            print(f"[Crawler] Error: {e}")
        _stop_event.wait(INTERVAL)


def _run_bm25():
    while not _stop_event.is_set():
        try:
            with _lock:
                with open(CRAWLED_PATH, "r", encoding="utf-8") as f:
                    pages_data = json.load(f)
            model = My_BM25(pages_data)
            _atomic_pkl(BM25_PATH, model)
        except Exception as e:
            print(f"[BM25] Error: {e}")
        _stop_event.wait(INTERVAL)


def _run_tfidf():
    while not _stop_event.is_set():
        try:
            with _lock:
                with open(CRAWLED_PATH, "r", encoding="utf-8") as f:
                    pages_data = json.load(f)
            model = My_TFIDF(pages_data)
            _atomic_pkl(TFIDF_PATH, model)
        except Exception as e:
            print(f"[TF-IDF] Error: {e}")
        _stop_event.wait(INTERVAL)


def _run_pagerank():
    while not _stop_event.is_set():
        try:
            with _lock:
                with open(CRAWLED_PATH, "r", encoding="utf-8") as f:
                    pages_data = json.load(f)
            graph = WebGraph(pages_data)
            scores = graph.calculate_pagerank()
            with _lock:
                _atomic_json(PAGERANK_PATH, scores)
        except Exception as e:
            print(f"[PageRank] Error: {e}")
        _stop_event.wait(INTERVAL)


_threads = []


def start_updater():
    _stop_event.clear()
    _threads.clear()
    for target in (_run_crawler, _run_bm25, _run_tfidf, _run_pagerank):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        _threads.append(t)


def stop_updater():
    _stop_event.set()
    for t in _threads:
        t.join(timeout=INTERVAL + 2)
