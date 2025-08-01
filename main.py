import os
import requests
import pypdf
import io
import time
import gc
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("INFO: Python script starting...")

# --- Configuration ---
try:
    print("INFO: Configuring Generative AI...")
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("INFO: Generative AI configured successfully.")
except KeyError:
    print("FATAL: GOOGLE_API_KEY environment variable not set.")
    exit(1)

# --- Load Models ---
print("INFO: Loading LLM...")
llm = genai.GenerativeModel('gemini-1.5-flash-latest')
print("INFO: LLM loaded successfully.")


# Limits for free-tier
MAX_PDF_SIZE = 1024 * 1024      # 1MB
MAX_PDF_PAGES = 10             # First 10 pages only
CHUNK_SIZE = 5                 # merge 5 paragraphs per chunk
MAX_CHUNKS = 8                 # max 8 chunks considered

class HackathonRequest(BaseModel):
    documents: str    # URL to PDF
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Hybrid RAG API")

def extract_text_from_pdf_url(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF too large (> {MAX_PDF_SIZE//1024} KB)")
        with io.BytesIO(response.content) as pdf_file:
            reader = pydp.PdfReader(pdf_file)
            pages = reader.pages[:MAX_PDF_PAGES]
            full_text = "".join(page.extract_text() or "" for page in pages)
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or read PDF: {e}")

def chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = ["\n\n".join(paragraphs[i:i+CHUNK_SIZE]) for i in range(0, len(paragraphs), CHUNK_SIZE)]
    return chunks[:MAX_CHUNKS]

def find_most_similar_chunk(chunks: List[str], question: str) -> str:
    if not chunks:
        return ""
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vecs = vectorizer.transform(chunks)
    question_vec = vectorizer.transform([question])
    sims = (chunk_vecs * question_vec.T).toarray().flatten()
    best_idx = int(np.argmax(sims))
    return chunks[best_idx] if sims[best_idx] > 0.1 else "" # Return context only if similarity is above a threshold

def generate_answer_with_llm(context: str, question: str) -> str:
    """
    Uses the Google Gemini model to generate a high-quality answer.
    """
    prompt = f"""Based only on the following context, please answer the question.
Do not use any external knowledge. If the answer is not in the context, say "The answer is not found in the provided context."

--- CONTEXT ---
{context}
--- END OF CONTEXT ---

QUESTION: {question}

ANSWER:
"""
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error processing the question with the language model."

@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    questions = request_body.questions
    full_text = extract_text_from_pdf_url(request_body.documents)
    chunks = chunk_text(full_text)
    
    if not chunks:
        # If no text, return a default answer for all questions
        return HackathonResponse(answers=["No text could be extracted from the document."] * len(questions))

    answers = []
    for question in questions:
        context = find_most_similar_chunk(chunks, question)
        if not context:
            answers.append("No relevant context was found in the document for this question.")
            continue
        
        answer = generate_answer_with_llm(context, question)
        answers.append(answer)

    # Cleanup to reduce RAM pressure
    del full_text, chunks
    gc.collect()

    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Hybrid RAG API is running!"}
