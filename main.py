import os
import requests
import io
import time
import gc
import google.generativeai as genai
import fitz  # PyMuPDF

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
MAX_PDF_SIZE = 2 * 1024 * 1024      # 2MB
MAX_PDF_PAGES = 12                 # First 12 pages only
CHUNK_SIZE = 1024                  # Characters per chunk
MAX_CHUNKS = 10                    # Up to 10 chunks per document

class HackathonRequest(BaseModel):
    documents: str    # URL to PDF
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Final Hybrid RAG API")

def extract_text_from_pdf_url(pdf_url: str) -> str:
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
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or read PDF: {e}")

def chunk_text(text: str) -> List[str]:
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    return chunks[:MAX_CHUNKS]

def find_most_similar_chunk(chunks: List[str], question: str) -> str:
    if not chunks:
        return ""
    all_text = chunks + [question]
    vectorizer = TfidfVectorizer().fit(all_text)
    tfidf_matrix = vectorizer.transform(all_text)
    chunk_vecs = tfidf_matrix[:-1]
    question_vec = tfidf_matrix[-1]
    sims = (chunk_vecs * question_vec.T).toarray().flatten()
    best_idx = int(np.argmax(sims))
    return chunks[best_idx] if sims[best_idx] > 0.05 else ""

def generate_answer_with_llm(context: str, question: str) -> str:
    """
    Uses the Google Gemini model to generate a high-quality answer.
    """
    prompt = f"""Based only on the following context, please answer the question accurately.
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
    t0 = time.time()
    questions = request_body.questions
    
    full_text = extract_text_from_pdf_url(request_body.documents)
    chunks = chunk_text(full_text)
    
    if not chunks:
        return HackathonResponse(answers=["No text could be extracted from the document."] * len(questions))

    answers = []
    for question in questions:
        context = find_most_similar_chunk(chunks, question)
        if not context:
            answers.append("No relevant context was found in the document for this question.")
            continue
        
        # Use the AI to generate the final answer
        answer = generate_answer_with_llm(context, question)
        answers.append(answer)

    del full_text, chunks
    gc.collect()

    print(f"Total processing time for {len(questions)} questions: {time.time()-t0:.2f}s")
    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Final Hybrid RAG API is running!"}
