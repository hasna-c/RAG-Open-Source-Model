# gemma2b_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import os
import time

PDF_PATH = "test.pdf"
OLLAMA_MODEL = "gemma:2b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma_db"

app = FastAPI()

# Load PDF and build vectorstore at startup
print("[*] Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

print("[*] Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"[*] Total chunks created: {len(chunks)}")

print("[*] Building vector store with embeddings...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

retriever = vectordb.as_retriever()
llm = OllamaLLM(model=OLLAMA_MODEL)
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    start_time = time.time()
    answer = qa.invoke({"query": query.question})
    inference_time = time.time() - start_time
    if isinstance(answer, dict):
        answer_text = answer.get("result", str(answer))
    else:
        answer_text = str(answer)
    return {"question": query.question, "answer": answer_text, "inference_time": inference_time}
