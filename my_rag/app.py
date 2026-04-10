import os
import streamlit as st
import requests

st.set_page_config(
    page_title="EU AI Act Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ EU AI Act Assistant")
st.caption("Ask any question about the EU Artificial Intelligence Act (Regulation 2024/1689)")

# Example questions
st.markdown("**Try asking:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("What are high-risk AI systems?"):
        st.session_state.question = "What are high-risk AI systems according to the EU AI Act?"
with col2:
    if st.button("What AI practices are prohibited?"):
        st.session_state.question = "What AI practices are explicitly prohibited under Article 5?"

# Input
question = st.text_input(
    "Your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What obligations do providers of high-risk AI systems have?"
)

if st.button("Ask") and question:
    with st.spinner("Searching the EU AI Act..."):
        try:
            api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
            response = requests.post(
            f"{api_url}/ask",
                json={"question": question}
            )
            data = response.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### Sources")
            st.caption(f"Retrieved from pages: {data['sources']}")

        except Exception as e:
            st.error("Make sure the API is running: uvicorn api:app --reload")

st.markdown("---")
st.caption("Built on EU AI Act (Regulation 2024/1689) | RAG pipeline: LangChain + FAISS + BM25 + Flashrank + Groq")