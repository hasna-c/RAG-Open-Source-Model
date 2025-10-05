# phi3_server.py

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- CONFIG ----------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = os.getenv("HF_TOKEN", None)
QUANTIZATION = "32bit"  # options: "8bit", "4bit", "fp16", "fp32"
PDF_PATH = r"C:\Users\vishnuu\Projects\RAG\test.pdf"

# ---------- RAG MODEL ----------
class RAGModel:
    def __init__(self, model_name=MODEL_NAME, hf_token=HF_TOKEN, quantization=QUANTIZATION):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RAGModel] Device: {self.device}")
        print(f"[RAGModel] Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

        print(f"[RAGModel] Loading model {model_name} with {quantization} quantization...")

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if quantization in ["fp16", "fp32"] else None
        }

        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=256):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9
        )
        return outputs[0]["generated_text"]


# ---------- RAG QA SYSTEM ----------
class RAGQA:
    def __init__(self, pdf_path=PDF_PATH):
        print("[RAGQA] Initializing generative model...")
        self.rag = RAGModel()

        print("[RAGQA] Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No documents found at {pdf_path}")

        print("[RAGQA] Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        print("[RAGQA] Building embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("[RAGQA] Building FAISS index...")
        self.vectorstore = FAISS.from_documents(docs, embeddings)

    def query(self, question):
        print(f"[RAGQA] Searching for: {question}")
        docs = self.vectorstore.similarity_search(question, k=3)
        if not docs:
            return "No relevant information found in the document."

        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
Answer the following question based ONLY on the retrieved document context.
Give a short, precise answer without extra explanation.

Question: {question}
Answer:
"""
        return self.rag.generate(prompt).strip()


# ---------- FASTAPI SERVER ----------
app = FastAPI(title="Phi-3 RAG QA Server", docs_url="/")

class Query(BaseModel):
    question: str

rag_system = RAGQA()

@app.post("/ask")
def ask_question(query: Query):
    answer = rag_system.query(query.question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phi3_server:app", host="127.0.0.1", port=8000, reload=False)
