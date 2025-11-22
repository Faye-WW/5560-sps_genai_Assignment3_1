from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from app.embeddings import (
    embed_word,
    embed_sentence,
    embed_batch_sentences,
    cosine_similarity,
)

app = FastAPI(title="SPS GenAI API", version="0.1.0")

@app.get("/")
def root():
    return {"message": "Hello, FastAPI with UV!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------- Embedding: word ----------
class WordReq(BaseModel):
    word: str = Field(..., description="Single word to return its embedding")

@app.post("/embed/word")
def embed_word_endpoint(req: WordReq):
    vec = embed_word(req.word)
    if not vec:
        raise HTTPException(400, "Empty token after parsing.")
    return {"word": req.word, "vector": vec}

# --------- Embedding: sentence ----------
class SentReq(BaseModel):
    text: Optional[str] = Field(None, description="Single sentence")
    texts: Optional[List[str]] = Field(None, description="Multiple sentences for batch processing")

@app.post("/embed/sentence")
def embed_sentence_endpoint(req: SentReq):
    if (req.text is None) and (req.texts is None):
        raise HTTPException(400, "Provide `text` or `texts`.")
    if req.text is not None:
        return {"vector": embed_sentence(req.text)}
    else:
        return {"vectors": embed_batch_sentences(req.texts)}

# --------- Similarity: words ----------
class WordSimReq(BaseModel):
    a: str = Field(..., description="Word A")
    b: str = Field(..., description="Word B")

@app.post("/similarity/words")
def similarity_words(req: WordSimReq):
    va = embed_word(req.a)
    vb = embed_word(req.b)
    return {"cosine_similarity": cosine_similarity(va, vb)}

# --------- Similarity: sentences ----------
class SentSimReq(BaseModel):
    a: str = Field(..., description="Sentence A")
    b: str = Field(..., description="Sentence B")

@app.post("/similarity/sentences")
def similarity_sentences(req: SentSimReq):
    va = embed_sentence(req.a)
    vb = embed_sentence(req.b)
    return {"cosine_similarity": cosine_similarity(va, vb)}

# classify/image
from fastapi import UploadFile, File
from app.inference import predict_image_bytes

@app.post("/classify/image")
def classify_image(file: UploadFile = File(...)):
    """
    Accept an image file (jpg/png) and return a predicted CIFAR-10 class.
    """
    content = file.file.read()
    result = predict_image_bytes(content)
    return {"filename": file.filename, **result}

# GAN routes
from fastapi import FastAPI 
from app.routers.gan import router as gan_router
app.include_router(gan_router) 


