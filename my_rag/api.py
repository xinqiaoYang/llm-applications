from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
import numpy as np
import os

app = FastAPI(title="EU AI Act Q&A API")

# ── startup: load everything once ──────────────────────────
print("Loading pipeline...")
loader = PyMuPDFLoader("eu_ai_act.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

corpus = [chunk.page_content for chunk in chunks]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt_template = """You are an expert on the EU AI Act.
Use the following context to answer accurately.
If you don't know, say so — never make things up.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

print("Pipeline ready!")

# ── input/output schemas ────────────────────────────────────
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[int]

# ── helper functions ────────────────────────────────────────
def expand_query(question):
    expansion_prompt = f"""You are an EU AI Act expert.
Rewrite this question using relevant legal terminology.
Do NOT invent article numbers you are not certain about.
Keep it under 2 sentences.

Question: {question}
Expanded query:"""
    return llm.invoke(expansion_prompt).content

def hybrid_retrieve(query, k_dense=10, k_bm25=10, k_final=3):
    dense_results = vectorstore.similarity_search(query, k=k_dense)

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    doc_scores = {}
    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        doc_scores[key] = doc_scores.get(key, 0) + 1/(rank + 60)
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        doc_scores[key] = doc_scores.get(key, 0) + 1/(rank + 60)

    all_docs = {doc.page_content: doc
                for doc in dense_results + bm25_results}
    sorted_docs = sorted(
        doc_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_docs = [all_docs[content] for content, _ in sorted_docs[:20]]

    passages = [{"id": i, "text": doc.page_content}
                for i, doc in enumerate(top_docs)]
    reranked = reranker.rerank(RerankRequest(query=query, passages=passages))

    final_indices = [r["id"] for r in reranked[:k_final]]
    return [top_docs[i] for i in final_indices]

# ── endpoints ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "EU AI Act RAG"}

@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    expanded = expand_query(payload.question)
    retrieved_docs = hybrid_retrieve(expanded)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    chain_prompt = prompt.format(context=context, question=payload.question)
    response = llm.invoke(chain_prompt)
    sources = [doc.metadata["page"] for doc in retrieved_docs]
    return Answer(
        question=payload.question,
        answer=response.content,
        sources=sources
    )