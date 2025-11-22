from typing import List, Union
import numpy as np
import spacy

# Load spaCy model with word vectors
_NLP = spacy.load("en_core_web_md")

def embed_word(word: str) -> List[float]:
    """
    Word embedding: return the vector of the first token.
    """
    doc = _NLP(word.strip())
    if len(doc) == 0:
        return []
    return doc[0].vector.astype(float).tolist()

def embed_sentence(text: str) -> List[float]:
    """
    Sentence embedding: use spaCy's doc.vector.
    """
    doc = _NLP(text)
    return doc.vector.astype(float).tolist()

def embed_batch_sentences(texts: List[str]) -> List[List[float]]:
    """
    Batch sentence embeddings using nlp.pipe.
    """
    vecs = []
    for doc in _NLP.pipe(texts):
        vecs.append(doc.vector.astype(float).tolist())
    return vecs

def cosine_similarity(a: Union[List[float], np.ndarray],
                      b: Union[List[float], np.ndarray]) -> float:
    """
    Cosine similarity between two vectors.
    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
