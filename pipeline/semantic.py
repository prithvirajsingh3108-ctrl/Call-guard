"""
pipeline/semantic.py
────────────────────
Layer 2: semantic similarity scoring using sentence-transformers.

SemanticScorer.score(text) returns the highest cosine similarity between
the input text and any phrase in phrases.json, plus the category and
nearest example phrase.

The model is loaded once (lazily) and phrase embeddings are cached to
disk so repeated runs don't re-encode the bank.

Falls back gracefully if sentence-transformers is not installed —
returns score=0.0 so the detector continues with keyword-only mode.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────

class SemanticResult(NamedTuple):
    score:          float   # 0.0 – 1.0  cosine similarity
    category:       str     # matched category or ""
    nearest_phrase: str     # the most similar phrase from the bank
    available:      bool    # False when sentence-transformers not installed


_UNAVAILABLE = SemanticResult(0.0, "", "", False)


class SemanticScorer:
    """
    Lazy-loading semantic similarity scorer.

    Parameters
    ----------
    model_name  : sentence-transformers model name
    phrases_path: path to phrases.json
    cache_dir   : directory for persisted phrase embeddings
    min_score   : return empty result when best score < this threshold
    """

    def __init__(
        self,
        model_name: str    = "paraphrase-multilingual-MiniLM-L12-v2",
        phrases_path: str  = "pipeline/phrases.json",
        cache_dir: str     = ".cache/semantic",
        min_score: float   = 0.45,
    ) -> None:
        self.model_name   = model_name
        self.phrases_path = phrases_path
        self.cache_dir    = Path(cache_dir)
        self.min_score    = min_score

        self._model       = None   # loaded on first use
        self._embeddings  = None   # {category: [(phrase, emb), ...]}
        self._available   = None   # True/False after first attempt

    # ── Lazy initialisation ───────────────────────────────────────────────

    def _load(self) -> bool:
        """Load model and phrase embeddings. Returns True on success."""
        if self._available is not None:
            return self._available

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("[semantic] sentence-transformers not installed — semantic layer disabled")
            self._available = False
            return False

        # Load phrase bank
        phrases_file = Path(self.phrases_path)
        if not phrases_file.exists():
            print(f"[semantic] phrases.json not found at {phrases_file} — semantic layer disabled")
            self._available = False
            return False

        with open(phrases_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        phrases: dict[str, list[str]] = {
            k: v for k, v in raw.items() if not k.startswith("_")
        }

        # Check disk cache (keyed by model name + phrases file mtime)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        mtime    = int(phrases_file.stat().st_mtime)
        safe_name = self.model_name.replace("/", "_").replace("-", "_")
        cache_file = self.cache_dir / f"{safe_name}_{mtime}.pkl"

        if cache_file.exists():
            print(f"[semantic] Loading cached phrase embeddings from {cache_file}")
            with open(cache_file, "rb") as f:
                self._embeddings = pickle.load(f)
            self._model = SentenceTransformer(self.model_name)
        else:
            print(f"[semantic] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print("[semantic] Encoding phrase bank …")
            self._embeddings = {}
            for cat, phrase_list in phrases.items():
                embs = self._model.encode(phrase_list, normalize_embeddings=True)
                self._embeddings[cat] = list(zip(phrase_list, embs))
            # Save cache
            with open(cache_file, "wb") as f:
                pickle.dump(self._embeddings, f)
            # Delete old cache files for this model
            for old in self.cache_dir.glob(f"{safe_name}_*.pkl"):
                if old != cache_file:
                    old.unlink(missing_ok=True)
            print("[semantic] Phrase embeddings cached")

        self._available = True
        return True

    # ── Public API ────────────────────────────────────────────────────────

    def score(self, text: str) -> SemanticResult:
        """
        Compute the maximum cosine similarity between `text` and all phrases.

        Returns a SemanticResult with score, category, nearest_phrase, available.
        Returns _UNAVAILABLE (score=0) if the model is not installed.
        """
        if not text or not text.strip():
            return SemanticResult(0.0, "", "", True)

        if not self._load():
            return _UNAVAILABLE

        try:
            import numpy as np
            query_emb = self._model.encode([text], normalize_embeddings=True)[0]

            best_score   = 0.0
            best_cat     = ""
            best_phrase  = ""

            for cat, pairs in self._embeddings.items():
                for phrase, emb in pairs:
                    sim = float(np.dot(query_emb, emb))
                    if sim > best_score:
                        best_score  = sim
                        best_cat    = cat
                        best_phrase = phrase

            if best_score < self.min_score:
                return SemanticResult(0.0, "", "", True)

            return SemanticResult(
                score          = round(best_score, 4),
                category       = best_cat,
                nearest_phrase = best_phrase,
                available      = True,
            )
        except Exception as exc:
            print(f"[semantic] Scoring error: {exc}")
            return SemanticResult(0.0, "", "", False)

    def invalidate_cache(self) -> None:
        """Delete all cached embeddings — call when phrases.json changes."""
        self._embeddings = None
        self._available  = None
        self._model      = None
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink(missing_ok=True)
        print("[semantic] Cache invalidated")


# ── Module-level singleton (shared across all imports) ────────────────────────
from config import cfg

_scorer: SemanticScorer | None = None


def get_scorer() -> SemanticScorer:
    global _scorer
    if _scorer is None:
        _scorer = SemanticScorer(
            model_name   = cfg.SEMANTIC_MODEL,
            phrases_path = cfg.PHRASES_PATH,
            cache_dir    = cfg.SEMANTIC_CACHE_DIR,
            min_score    = cfg.SEMANTIC_MIN_SCORE,
        )
    return _scorer


def semantic_score(text: str) -> SemanticResult:
    """Convenience wrapper — use this in the detector."""
    if not cfg.SEMANTIC_ENABLED:
        return _UNAVAILABLE
    return get_scorer().score(text)
