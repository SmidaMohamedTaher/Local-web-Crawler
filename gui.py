import json
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from search.search import Search
from indexer.BM25 import My_BM25
from indexer.TF_IDF import My_TFIDF

from search.search import Search


# ── colours & fonts ──────────────────────────────────────────────────────────
BG        = "#1e1e2e"
SURFACE   = "#2a2a3d"
ACCENT    = "#7c6af7"
ACCENT_H  = "#9d8fff"
TEXT      = "#e0e0f0"
SUBTEXT   = "#9090b0"
CARD_BG   = "#252538"
CARD_BD   = "#3a3a55"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_LABEL  = ("Segoe UI", 10)
FONT_ENTRY  = ("Segoe UI", 12)
FONT_BTN    = ("Segoe UI", 11, "bold")
FONT_CARD_T = ("Segoe UI", 11, "bold")
FONT_CARD_S = ("Segoe UI", 9)

METHODS = ["TF-IDF", "BM25", "TF-IDF + PageRank", "BM25 + PageRank"]


def load_models():
    tfidf    = joblib.load("APIs/tfidf_model.pkl")
    bm25     = joblib.load("APIs/bm25_model.pkl")
    with open("APIs/pagerank_scores.json", "r", encoding="utf-8") as f:
        pagerank = json.load(f)
    return Search(tfidf, bm25, pagerank)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Web Crawler — Search Engine")
        self.geometry("780x620")
        self.minsize(640, 480)
        self.configure(bg=BG)

        self.search_engine = load_models()
        self._build_ui()

       
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # header
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=30, pady=(28, 0))

        tk.Label(header, text="🔍 Local Search Engine",
                 font=FONT_TITLE, bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(header, text="Search across your local HTML pages",
                 font=FONT_LABEL, bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(2, 0))

        # search bar row
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=30, pady=(18, 0))

        self._entry = tk.Entry(bar, font=FONT_ENTRY, bg=SURFACE, fg=TEXT,
                               insertbackground=TEXT, relief="flat",
                               highlightthickness=1, highlightbackground=CARD_BD,
                               highlightcolor=ACCENT)
        self._entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self._entry.bind("<Return>", lambda _: self._do_search())

        self._method_var = tk.StringVar(value=METHODS[0])
        method_menu = ttk.Combobox(bar, textvariable=self._method_var,
                                   values=METHODS, state="readonly",
                                   font=FONT_LABEL, width=20)
        method_menu.pack(side="left", ipady=6, padx=(0, 10))

        btn = tk.Button(bar, text="Search", font=FONT_BTN,
                        bg=ACCENT, fg="#ffffff", activebackground=ACCENT_H,
                        activeforeground="#ffffff", relief="flat",
                        padx=18, pady=6, cursor="hand2",
                        command=self._do_search)
        btn.pack(side="left")

        # status label
        self._status = tk.Label(self, text="", font=FONT_LABEL,
                                bg=BG, fg=SUBTEXT)
        self._status.pack(anchor="w", padx=30, pady=(6, 0))

        # results area
        results_frame = tk.Frame(self, bg=BG)
        results_frame.pack(fill="both", expand=True, padx=30, pady=(10, 20))

        canvas = tk.Canvas(results_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical",
                                  command=canvas.yview)
        self._cards_frame = tk.Frame(canvas, bg=BG)

        self._cards_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._cards_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    # ── search logic ─────────────────────────────────────────────────────────

    def _do_search(self):
        query = self._entry.get().strip()
        if not query:
            return

        # reload models so results reflect latest background update

        method = self._method_var.get()
        dispatch = {
            "TF-IDF":             self.search_engine.search_tfidf,
            "BM25":               self.search_engine.search_bm25,
            "TF-IDF + PageRank":  self.search_engine.search_tfidf_pagerank,
            "BM25 + PageRank":    self.search_engine.search_bm25_pagerank,
        }
        results = dispatch[method](query)
        self._render_results(results, method)

    def _render_results(self, results, method):
        for widget in self._cards_frame.winfo_children():
            widget.destroy()

        if not results:
            tk.Label(self._cards_frame, text="No results found.",
                     font=FONT_LABEL, bg=BG, fg=SUBTEXT).pack(pady=20)
            self._status.config(text="0 results")
            return

        self._status.config(text=f"{len(results)} result(s) — {method}")

        for i, r in enumerate(results, 1):
            card = tk.Frame(self._cards_frame, bg=CARD_BG,
                            highlightthickness=1, highlightbackground=CARD_BD)
            card.pack(fill="x", pady=(0, 8))

            inner = tk.Frame(card, bg=CARD_BG)
            inner.pack(fill="x", padx=14, pady=10)

            tk.Label(inner, text=f"{i}. {r['title']}",
                     font=FONT_CARD_T, bg=CARD_BG, fg=TEXT,
                     anchor="w").pack(fill="x")
            tk.Label(inner, text=r["url"],
                     font=FONT_CARD_S, bg=CARD_BG, fg=ACCENT,
                     anchor="w").pack(fill="x")
            tk.Label(inner, text=f"Score: {r['score']:.4f}",
                     font=FONT_CARD_S, bg=CARD_BG, fg=SUBTEXT,
                     anchor="w").pack(fill="x")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
