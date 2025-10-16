import os
import io
import pickle
import re
import requests
from typing import List, Dict, Tuple
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from PIL import Image
import pytesseract
from offence_rules import detect_offence, extract_fine_amount

# -------------------- Setup --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
INDEX_DIR = "embeddings"
os.makedirs(INDEX_DIR, exist_ok=True)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Legal Chatbot & Challan Analyzer", layout="wide")
st.title("🧑‍⚖️ Indian Legal Chatbot & Challan Analyzer")

# -------------------- Helper Functions --------------------
@st.cache_resource
def load_embedding_model(name: str = EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

def pdf_to_text(file_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(file_bytes))
    return [p.extract_text() or "" for p in reader.pages]

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        if end >= len(text): break
        start = end - overlap
    return chunks

def embed_texts(model, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    return embs

def build_faiss_index(embs: np.ndarray):
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def save_index(index, metadata, name):
    faiss.write_index(index, f"{INDEX_DIR}/{name}.index")
    with open(f"{INDEX_DIR}/{name}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_index(name):
    index = faiss.read_index(f"{INDEX_DIR}/{name}.index")
    with open(f"{INDEX_DIR}/{name}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve(index, meta, query_emb, k=4):
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(meta):
            results.append({**meta[idx], "score": float(score)})
    return results

def ollama_call(prompt):
    try:
        payload = {"model": "phi3", "prompt": prompt, "stream": False}
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
        return r.json().get("response", "")
    except Exception as e:
        return f"[Ollama error: {e}]"

# -------------------- Combined PDF + Image Uploader --------------------
st.header("📄 Upload Legal PDFs or 📸 Challan Images")

uploaded_files = st.file_uploader(
    "Upload your legal PDFs and/or traffic challan images (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

model = load_embedding_model()

if uploaded_files:
    for file in uploaded_files:
        file_ext = file.name.lower().split(".")[-1]

        # -------------------- PDF Upload --------------------
        if file_ext == "pdf":
            with st.spinner(f"Processing PDF: {file.name}"):
                pages = pdf_to_text(file.read())
                chunks, meta = [], []
                for pnum, text in enumerate(pages, 1):
                    for cnum, chunk in enumerate(chunk_text(text)):
                        chunks.append(chunk)
                        meta.append({
                            "source_id": f"{file.name}_p{pnum}_c{cnum}",
                            "page_number": pnum,
                            "filename": file.name,
                            "text": chunk
                        })
                embs = embed_texts(model, chunks)
                index = build_faiss_index(embs)
                prefix = os.path.splitext(file.name)[0]
                save_index(index, meta, prefix)
                st.success(f"✅ Indexed {len(chunks)} chunks → prefix: {prefix}")

        # -------------------- Challan Upload --------------------
        elif file_ext in ["jpg", "jpeg", "png"]:
            with st.spinner(f"Analyzing Challan: {file.name}"):
                img = Image.open(file).convert("RGB")
                st.image(img, caption=file.name, use_container_width=True)
                text = pytesseract.image_to_string(img)

                fine = extract_fine_amount(text)
                offence_info = detect_offence(text)

                offence = offence_info["offence"]
                section = offence_info["section"]
                advice = offence_info["advice"]
                fine_text = offence_info["fine"] if fine == "Not mentioned" else f"₹{fine}"

                st.markdown("### 👮 Your Challan Explained (Simple Words)")
                st.markdown(f"**🚦 Violation:** {offence}")
                st.markdown(f"**💰 Fine:** {fine_text}")
                st.markdown(f"**📜 Section:** {section}")
                st.markdown(f"**✅ What To Do:** {advice}")

# -------------------- Step 2: Ask Questions --------------------
st.header("🤖 Ask a Question about Your Uploaded PDFs or Challan")

index_prefix = st.text_input("Enter the index prefix (e.g., Traffic_Law)")
question = st.text_area("Type your question here (e.g., 'What is the fine for overspeeding?')")

if st.button("Get Answer"):
    if not index_prefix:
        st.warning("⚠️ Please enter an index prefix.")
    elif not question:
        st.warning("⚠️ Please enter your question.")
    else:
        try:
            index, meta = load_index(index_prefix)
        except Exception as e:
            st.error(f"❌ Could not load index: {e}")
            st.stop()

        model = load_embedding_model()
        q_emb = embed_texts(model, [question])
        retrieved = retrieve(index, meta, q_emb, 3)

        st.subheader("📘 Retrieved Context (Top Matches)")
        for r in retrieved:
            st.write(f"**Page {r.get('page_number')}** — {r.get('text')[:500]}...")

        # Build simple, human-friendly LLM prompt
        context = "\n".join([r.get("text") for r in retrieved])
        prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Explain the answer in simple, friendly language that a normal person can understand. "
            "Avoid legal jargon and keep it concise with examples if helpful."
        )

        with st.spinner("💬 Generating answer..."):
            answer = ollama_call(prompt)

        st.subheader("🧠 Simple Explanation")
        st.write(answer)

# -------------------- Sidebar --------------------
st.sidebar.header("ℹ️ Indexed Files")
pdfs = [f for f in os.listdir(INDEX_DIR) if f.endswith(".index")]
if pdfs:
    st.sidebar.success(f"Available Indexes: {len(pdfs)}")
    for p in pdfs:
        st.sidebar.write(f"- {p[:-6]}")
else:
    st.sidebar.warning("No PDFs indexed yet.")

st.sidebar.markdown("---")
st.sidebar.write("Model: **Phi-3 (Ollama)** — running locally")
st.sidebar.write("Language: Simple Indian English 🇮🇳")
st.sidebar.markdown("Built for citizens to easily understand Indian laws & traffic fines ⚖️")
