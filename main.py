import os
import requests
import io
import time
import gc

import fitz  # PyMuPDF (very fast PDF extraction)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("INFO: Python script starting...")

# ---- TUNABLE LIMITS ----
MAX_PDF_SIZE = 2 * 1024 * 1024  # 2MB
MAX_PDF_PAGES = 12             # First 12 pages
CHUNK_SIZE = 1024              # Characters per chunk
MAX_CHUNKS = 10                # Up to 10 chunks per document

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Optimized Fast RAG (PyMuPDF + TF-IDF)")

def extract_text_from_pdf_url(pdf_url: str) -> str:
    t0 = time.time()
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF too large (> {MAX_PDF_SIZE//1024} KB)")
        with io.BytesIO(response.content) as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            texts = []
            for page in doc.pages(0, min(len(doc), MAX_PDF_PAGES)):
                texts.append(page.get_text())
            full_text = "\n\n".join(texts)
        print(f"PDF extracted in {time.time()-t0:.2f}s")
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or read PDF: {e}")

def chunk_text(text: str) -> List[str]:
    # Fixed-size chunking by chars for fast, even splitting
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    return chunks[:MAX_CHUNKS]

def simple_qa(context: str, question: str) -> str:
    # Simple QA: pick the sentence in the context with most keyword overlap, else fallback
    key_words = [w.strip(',.?!";:').lower() for w in question.split() if len(w) > 3]
    sentences = context.replace('\n', ' ').split('.')
    best_score, best_sent = -1, ""
    for sent in sentences:
        score = sum(1 for word in key_words if word in sent.lower())
        if score > best_score and score > 0:
            best_score, best_sent = score, sent.strip()
    if best_sent:
        return best_sent
    return "The answer is not found in the provided context."

@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    t0 = time.time()
    questions = request_body.questions
    
    # Step 1 & 2: Extract text and chunk it
    full_text = extract_text_from_pdf_url(request_body.documents)
    chunks = chunk_text(full_text)
    
    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from PDF.")

    # Step 3: Vectorize all chunks and questions together (much faster)
    vectorizer = TfidfVectorizer().fit(chunks + questions)
    chunk_vecs = vectorizer.transform(chunks)
    question_vecs = vectorizer.transform(questions)

    # Step 4: Calculate all similarities at once using matrix multiplication
    similarity_matrix = cosine_similarity(question_vecs, chunk_vecs)

    answers = []
    for i in range(len(questions)):
        # Find the best chunk for the current question from the similarity matrix
        best_chunk_index = np.argmax(similarity_matrix[i])
        
        # Check if the similarity score is above a certain threshold
        if similarity_matrix[i][best_chunk_index] > 0.05:
            context = chunks[best_chunk_index]
            answer = simple_qa(context, questions[i])
            answers.append(answer)
        else:
            answers.append("The answer is not found in the provided context.")

    # Free memory right after processing
    del full_text, chunks, vectorizer, chunk_vecs, question_vecs, similarity_matrix
    gc.collect()

    print(f"Total processing time for {len(questions)} questions: {time.time()-t0:.2f}s")
    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Optimized Fast RAG API is running!"}
