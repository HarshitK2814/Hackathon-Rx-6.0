import os
import requests
import pypdf
import io
import google.generativeai as genai
import chromadb

# Add a print statement right at the beginning
print("INFO: Python script starting...")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

print("INFO: Imports successful.")

# --- Configuration ---
try:
    print("INFO: Configuring Generative AI...")
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("INFO: Generative AI configured successfully.")
except KeyError:
    print("FATAL: GOOGLE_API_KEY environment variable not set.")
    exit(1)


# --- Load Models ---
print("INFO: Loading embedding model (this may take a while)...")
embedding_model = SentenceTransformer('paraphrase-albert-small-v2')
print("INFO: Embedding model loaded successfully.")

print("INFO: Loading LLM...")
llm = genai.GenerativeModel('gemini-1.5-flash-latest')
print("INFO: LLM loaded successfully.")


# --- Pydantic Models ---
class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


class HackathonResponse(BaseModel):
    answers: List[str]


# --- FastAPI App ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Bot",
    description="An advanced RAG API to answer questions from documents."
)


# --- Core RAG Logic ---
def answer_questions_from_document(document_url: str, questions: List[str]) -> List[str]:
    """
    Downloads a PDF, builds a vector index, and answers questions using RAG.
    """
    # 1. Download and Extract Text from PDF
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        with io.BytesIO(response.content) as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            full_document_text = "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # 2. Chunk the Text
    # Simple chunking by splitting text by double newlines.
    text_chunks = [p.strip() for p in full_document_text.split("\n\n") if p.strip()]
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Could not extract text chunks from document.")

    # 3. Create Vector Index in Memory (using ChromaDB)
    client = chromadb.Client()  # In-memory client by default
    collection = client.create_collection(name="document_chunks")

    # Generate embeddings and add to collection
    chunk_embeddings = embedding_model.encode(text_chunks).tolist()
    collection.add(
        embeddings=chunk_embeddings,
        documents=text_chunks,
        ids=[f"chunk_{i}" for i in range(len(text_chunks))]
    )

    # 4. Process Each Question
    final_answers = []
    for question in questions:
        # Find relevant chunks for the current question
        question_embedding = embedding_model.encode([question]).tolist()
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=3  # Get the top 3 most relevant chunks
        )

        relevant_context = "\n\n".join(results['documents'][0])

        # 5. Generate Answer with LLM (Augmented Generation)
        prompt = f"""Based only on the following context, please answer the question.
Do not use any external knowledge. If the answer is not in the context, say "The answer is not found in the provided context."

--- CONTEXT ---
{relevant_context}
--- END OF CONTEXT ---

QUESTION: {question}

ANSWER:
"""

        try:
            response = llm.generate_content(prompt)
            final_answers.append(response.text.strip())
        except Exception as e:
            print(f"LLM Error: {e}")
            final_answers.append("Error processing the question with the language model.")

    return final_answers


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    answers = answer_questions_from_document(
        document_url=request_body.documents,
        questions=request_body.questions
    )
    return HackathonResponse(answers=answers)


@app.get("/")
def read_root():
    return {"status": "Intelligent RAG API is running!"}
