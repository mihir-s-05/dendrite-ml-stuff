"""Phase-0 data: three genuinely distinct text domains, BPE-tokenized.

  - wiki   : Salesforce/wikitext (wikitext-2-raw-v1)  -> encyclopedic English
  - stories: roneneldan/tiny_stories                  -> simple narrative English
  - code   : CPython source (downloaded)              -> source code

The first two load offline from the local HF cache; code is downloaded with a
synthetic fallback. Tokenized with tiktoken gpt2 BPE (byte-level fallback).
Encoded token arrays are cached to data/*.npy so repeated runs are fast.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

CODE_URLS = [
    "https://raw.githubusercontent.com/python/cpython/3.12/Lib/_pydecimal.py",
    "https://raw.githubusercontent.com/python/cpython/3.12/Lib/argparse.py",
    "https://raw.githubusercontent.com/python/cpython/3.12/Lib/statistics.py",
]


# ------------------------------ tokenizer ------------------------------------
def get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return "bpe", enc.n_vocab, (lambda s: enc.encode_ordinary(s))
    except Exception as e:
        print(f"  [warn] tiktoken unavailable ({e}); byte-level tokens.")
        return "byte", 256, (lambda s: list(s.encode("utf-8", errors="ignore")))


# ------------------------------ sources --------------------------------------
def _hf_text(name, config, split, max_chars):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split)
    parts, total = [], 0
    for row in ds:
        t = row.get("text", "")
        if not t:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "".join(parts)[:max_chars]


def _code(max_chars):
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, "code.txt")
    if os.path.exists(cache):
        return open(cache, encoding="utf-8", errors="ignore").read()[:max_chars]
    chunks = []
    for url in CODE_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                chunks.append(r.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"  [warn] code download failed for {url} ({e}).")
    text = "\n\n".join(chunks)
    if len(text) < 1000:
        rng = np.random.default_rng(0)
        lines = ["def f(x):\n    return x * 2\n", "for i in range(10):\n    print(i)\n",
                 "class A:\n    def m(self):\n        return self.v\n"]
        text = "".join(str(rng.choice(lines)) for _ in range(max_chars // 20))
    else:
        open(cache, "w", encoding="utf-8").write(text)
    return text[:max_chars]


def _load_text(name, max_chars):
    try:
        if name == "wiki":
            return _hf_text("Salesforce/wikitext", "wikitext-2-raw-v1", "train", max_chars)
        if name == "stories":
            return _hf_text("roneneldan/tiny_stories", None, "train", max_chars)
        if name == "code":
            return _code(max_chars)
    except Exception as e:
        print(f"  [warn] {name}: load failed ({e}); skipping to synthetic code.")
        return _code(max_chars)


# ------------------------------ public API -----------------------------------
def load_domains(max_chars: int = 1_500_000, domains=("wiki", "stories", "code")):
    """Return (data, vocab_size, tok_kind).

    data: dict[name] -> (train_ids np.int64, val_ids np.int64).

    Token arrays are cached to data/*.npy. NOTE: this imports `datasets` /
    pyarrow only when a cache is missing; on Windows that must NOT happen in a
    process that has already imported torch (DLL clash). Run this module as a
    script first (`python src/phase0_data.py`) to materialize the caches, then
    the torch-side reads numpy only.
    """
    kind, vocab, encode = get_tokenizer()
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {}
    for name in domains:
        cache = os.path.join(DATA_DIR, f"tok_{name}_{kind}_{max_chars}.npy")
        if os.path.exists(cache):
            ids = np.load(cache)
        else:
            text = _load_text(name, max_chars)
            ids = np.array(encode(text), dtype=np.int64)
            np.save(cache, ids)
        n = int(0.9 * len(ids))
        data[name] = (ids[:n].copy(), ids[n:].copy())
        print(f"  {name:10s} tokens={len(ids):>9d}  (vocab={vocab}, {kind})")
    return data, vocab, kind


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pre-build tokenized domain caches.")
    ap.add_argument("--max-chars", type=int, default=1_500_000)
    a = ap.parse_args()
    print("Building Phase-0 token caches...")
    load_domains(max_chars=a.max_chars)
    print("Done.")
