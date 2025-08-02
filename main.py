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
from sklearn.metrics.pairwise import cosine_similarity
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


# ---- AGGRESSIVE LIMITS FOR FREE TIER ----
MAX_PDF_SIZE = 2 * 1024 * 1024      # 2MB
MAX_PDF_PAGES = 16                 # First 16 pages ONLY
CHUNK_SIZE = 1024                  # Characters per chunk
MAX_CHUNKS = 12                    # Max 12 chunks to keep TF-IDF fast

class HackathonRequest(BaseModel):
    documents: str
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

    # Vectorize all chunks and questions together for speed
    vectorizer = TfidfVectorizer().fit(chunks + questions)
    chunk_vecs = vectorizer.transform(chunks)
    question_vecs = vectorizer.transform(questions)

    # Calculate all similarities at once
    similarity_matrix = cosine_similarity(question_vecs, chunk_vecs)

    answers = []
    for i in range(len(questions)):
        # Get similarity scores for the current question
        scores = similarity_matrix[i]
        
        # Get the indices of the top 2 chunks, sorted from best to worst
        top_k_indices = np.argsort(scores)[-2:][::-1]
        
        # Combine the top chunks into a single context
        context_parts = []
        for index in top_k_indices:
            # Only include chunks that have some relevance
            if scores[index] > 0.05:
                context_parts.append(chunks[index])
        
        if context_parts:
            # Join the relevant chunks with a separator to give the LLM more info
            context = "\n\n---\n\n".join(context_parts)
            answer = generate_answer_with_llm(context, questions[i])
            answers.append(answer)
        else:
            answers.append("No relevant context was found in the document for this question.")

    del full_text, chunks, vectorizer, chunk_vecs, question_vecs, similarity_matrix
    gc.collect()

    print(f"Total processing time for {len(questions)} questions: {time.time()-t0:.2f}s")
    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Final Hybrid RAG API is running!"}
