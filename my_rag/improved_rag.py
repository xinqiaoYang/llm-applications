import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
import numpy as np

print("Loading PDF...")
loader = PyMuPDFLoader("eu_ai_act.pdf")
documents = loader.load()

print("Chunking with size=300...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Embedding and indexing...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("Building BM25 index...")
corpus = [chunk.page_content for chunk in chunks]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print("Loading reranker...")
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt_template = """You are an expert on the EU AI Act.
Use the following context to answer the question accurately.
If you don't know, say so — never make things up.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# def expand_query(question):
#     """Use LLM to expand query with relevant legal terms"""
#     expansion_prompt = f"""You are an EU AI Act expert.
# Rewrite this question to include specific legal terms,
# article numbers, or annexes that would help find the answer.
# Keep it under 2 sentences.

# Question: {question}
# Expanded query:"""
#     response = llm.invoke(expansion_prompt)
#     expanded = response.content
#     print(f"Expanded query: {expanded}")
#     return expanded

def expand_query(question):
    """Use LLM to expand query with relevant legal terms"""
    expansion_prompt = f"""You are an EU AI Act expert.
Rewrite this question using relevant legal terminology from the EU AI Act.
Do NOT invent article numbers or annex references you are not certain about.
Focus on legal concepts and terminology only.
Keep it under 2 sentences.

Question: {question}
Expanded query:"""
    response = llm.invoke(expansion_prompt)
    expanded = response.content
    print(f"Expanded query: {expanded}")
    return expanded


def hybrid_retrieve(query, k_dense=10, k_bm25=10, k_final=3):
    """
    Step 1: Retrieve top-10 with FAISS (semantic/dense search)
    Step 2: Retrieve top-10 with BM25 (keyword/lexical search)
    Step 3: Combine with Reciprocal Rank Fusion
    Step 4: Rerank with cross-encoder
    Step 5: Return top-3
    """
    # Dense retrieval
    dense_results = vectorstore.similarity_search(query, k=k_dense)

    # BM25 retrieval
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    # Reciprocal Rank Fusion
    doc_scores = {}
    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        doc_scores[key] = doc_scores.get(key, 0) + 1/(rank + 60)

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        doc_scores[key] = doc_scores.get(key, 0) + 1/(rank + 60)

    # Sort by RRF score
    all_docs = {doc.page_content: doc
                for doc in dense_results + bm25_results}
    sorted_docs = sorted(
        doc_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_docs = [all_docs[content] for content, _ in sorted_docs[:20]]

    # Rerank with cross-encoder
    passages = [{"id": i, "text": doc.page_content}
                for i, doc in enumerate(top_docs)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = reranker.rerank(rerank_request)

    # Return top-k final
    final_indices = [r["id"] for r in reranked[:k_final]]
    return [top_docs[i] for i in final_indices]


def ask(question):
    print(f"\nQ: {question}")

    # Expand query with legal terms
    expanded = expand_query(question)

    # Hybrid retrieval on expanded query
    retrieved_docs = hybrid_retrieve(expanded)

    # Build context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Generate answer using original question
    chain_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(chain_prompt)

    print(f"A: {response.content}")
    print(f"Sources: Pages {[doc.metadata['page'] for doc in retrieved_docs]}")
    print("-" * 60)


print("\n--- IMPROVED RAG: chunk=300 + Hybrid + Reranking + Query Expansion ---\n")

ask("What are high-risk AI systems according to the EU AI Act?")
ask("What obligations do providers of high-risk AI systems have?")
ask("What is prohibited AI under the EU AI Act?")