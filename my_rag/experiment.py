import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load document once
loader = PyMuPDFLoader("eu_ai_act.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Load embeddings once — reused across all experiments
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# Test question
test_question = "What are high-risk AI systems according to the EU AI Act?"

# Results storage
results = []

for chunk_size in [200, 300, 500, 1000]:
    print(f"\nRunning experiment: chunk_size={chunk_size}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    result = qa.invoke({"query": test_question})

    results.append({
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "answer": result["result"],
        "source_page": result["source_documents"][0].metadata["page"]
    })

# Print comparison
print("\n" + "="*60)
print("CHUNK SIZE EXPERIMENT RESULTS")
print("="*60)
print(f"Question: {test_question}\n")

for r in results:
    print(f"Chunk size: {r['chunk_size']} | Total chunks: {r['num_chunks']}")
    print(f"Source page: {r['source_page']}")
    print(f"Answer: {r['answer'][:400]}...")
    print("-"*60)