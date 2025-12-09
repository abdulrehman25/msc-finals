from typing import List
import numpy as np

def build_sessions(vectors: List, metas: List, window: int=20, step: int=10):
    sequences = []
    key_to_seq = {}
    for v, m in zip(vectors, metas):
        if v is None:
            continue
        key = (m.get("ip",""), m.get("ua",""))
        key_to_seq.setdefault(key, []).append(v)
    for _, seq in key_to_seq.items():
        if not seq or len(seq) < window:
            continue
        seq = np.stack(seq)
        i = 0
        while i + window <= len(seq):
            sequences.append(seq[i:i+window])
            i += step
    if not sequences:
        return np.zeros((0, window, vectors[0].shape[-1]), dtype=np.float32)
    return np.stack(sequences)
