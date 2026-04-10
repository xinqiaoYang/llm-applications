import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load
print("Loading PDF...")
loader = PyMuPDFLoader("eu_ai_act.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Chunk
print("Chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Embed + Index
print("Embedding and indexing (this takes 1-2 mins)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Index ready!")

# 4. LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# 5. Prompt template
prompt_template = """You are an expert on the EU AI Act.
Use the following context to answer the question accurately.
If you don't know the answer, say so — don't make things up.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 6. Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 7. Ask questions
print("\n--- EU AI Act Q&A System Ready ---\n")

questions = [
    "What is the main purpose of the EU AI Act?",
    "What are high-risk AI systems according to the EU AI Act?",
    "What obligations do providers of high-risk AI systems have?",
]

for q in questions:
    print(f"Q: {q}")
    result = qa_chain.invoke({"query": q})
    print(f"A: {result['result']}")
    print(f"Source: Page {result['source_documents'][0].metadata['page']}")
    print("-" * 50)