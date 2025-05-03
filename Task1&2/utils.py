import os, json, re
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Constants
PROFANITY_WORDS = ["asshole", "bastard", "bitch", "damn", "fuck", "hell", "piss", "shit", "crap", "jerk"]
PROF_PATTERN   = re.compile(r"\b(" + "|".join(PROFANITY_WORDS) + r")\b", re.IGNORECASE)
SENSITIVE      = ["balance", "owe", "debt", "account", "owing", "due", "statement"]
VERIFY         = ["date of birth", "dob", "address", "social security", "ssn"]

# Choose your embedding model: high-accuracy vs high-speed
_EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")  # or "all-MiniLM-L6-v2"

def load_json_convs(files) -> Dict[str, List[dict]]:
    convs = {}
    for f in files:
        cid = os.path.splitext(f.name)[0]
        convs[cid] = json.load(f)
    return convs

def annotate_with_similarity_orderaware(convs: Dict[str, List[dict]],
                                        prof_threshold: float = 0.6,
                                        sens_threshold: float = 0.6
                                       ) -> List[dict]:
    """
    Returns a list of records with 'text','profanity','privacy' labels.
    Privacy=1 only if the utterance is similar to SENSITIVE *and* occurs before any VERIFY utterance.
    """
    # Flatten utterances and keep track of call/utterance index
    all_texts, meta = [], []
    for cid, utts in convs.items():
        for idx, utt in enumerate(utts):
            all_texts.append(utt["text"])
            meta.append((cid, idx, utt["speaker"].lower(), utt["text"]))
    N = len(all_texts)

    # Embed
    embeddings = _EMBED_MODEL.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    def get_match_indices(seeds: List[str], threshold: float) -> set:
        seed_emb = _EMBED_MODEL.encode(seeds, convert_to_numpy=True)
        faiss.normalize_L2(seed_emb)
        D, I = index.search(seed_emb, k=N)
        hits = set()
        for d_row, i_row in zip(D, I):
            for dist, i in zip(d_row, i_row):
                if dist >= threshold:
                    hits.add(i)
        return hits

    prof_hits  = get_match_indices(PROFANITY_WORDS, prof_threshold)
    sens_hits  = get_match_indices(SENSITIVE, sens_threshold)
    verify_hits= get_match_indices(VERIFY, sens_threshold)

    # Group verify indices by call
    verify_by_call = {}
    for i in verify_hits:
        cid, idx, *_ = meta[i]
        verify_by_call.setdefault(cid, []).append(idx)

    # Build annotated records
    records = []
    for i, (cid, idx, speaker, text) in enumerate(meta):
        # Profanity: regex OR semantic
        prof = int(bool(PROF_PATTERN.search(text)) or (i in prof_hits))

        # Privacy (only for agents): semantic AND order-aware
        if speaker.startswith("agent") and (i in sens_hits):
            # If this call has any verify utterance later than idx?
            later_verifies = [v for v in verify_by_call.get(cid, []) if v > idx]
            priv = int(len(later_verifies) == 0)  # violation if NO verification after this utterance
        else:
            priv = 0

        records.append({"text": text, "profanity": prof, "privacy": priv})

    return records
