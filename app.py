import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.utilities import WikipediaAPIWrapper

# -------------------------
# 1. CONFIGURATION
# -------------------------
load_dotenv()

# Initialize chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for API Key
st.sidebar.subheader("🔑 API Settings")
user_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

llm = None
if user_api_key:
    llm = ChatGroq(
        groq_api_key=user_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

# -------------------------
# 2. FILE UPLOAD (RAG DATA)
# -------------------------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload your knowledge file (txt, csv, pdf, docx)",
    type=["txt", "csv", "pdf", "docx"]
)

vectorstore = None
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(raw_text)
    documents = [Document(page_content=x) for x in docs]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

# -------------------------
# 3. TOOLS
# -------------------------
def globalmart_rag(query: str) -> str:
    """Fetch relevant answers from the uploaded GlobalMart knowledge file using RAG."""
    if not vectorstore:
        return "⚠️ Please upload a knowledge file first."
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(query)
    if not results:
        return "No relevant info found in uploaded file."
    return "\n".join([doc.page_content for doc in results])

def calc(query: str) -> str:
    """Perform basic arithmetic calculations based on the query string."""
    try:
        return str(eval(query))
    except Exception:
        return "❌ Calculation error"

def wiki_search(query: str) -> str:
    """Search Wikipedia for general knowledge queries."""
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

# -------------------------
# 4. ROUTER (Rule-based)
# -------------------------
def route_query(query: str) -> dict:
    """Simple rule-based router that decides which tool(s) to use."""
    query_lower = query.lower()

    if any(word in query_lower for word in ["calculate", "sum", "multiply", "divide", "add", "minus", "+", "-", "*", "/"]):
        return {"tool": "Calculator", "result": calc(query)}

    elif any(word in query_lower for word in ["company", "policy", "globalmart", "sales", "internal"]):
        return {"tool": "GlobalMart RAG", "result": globalmart_rag(query)}

    else:
        return {"tool": "Wikipedia", "result": wiki_search(query)}

# -------------------------
# 5. STREAMLIT UI
# -------------------------
st.title("🛠 Multi-Tool Agent (Groq + File RAG + Wikipedia + Calculator)")

query = st.text_area("💬 Ask your question:")

if st.button("Run Assistant"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    elif not llm:
        st.error("❌ Please enter your API key in the sidebar first.")
    else:
        with st.spinner("🤖 Thinking..."):
            # Step 1: Route to tool
            routed = route_query(query)

            # Step 2: Use Groq LLM to synthesize final answer
            synthesis_prompt = f"""
            You are an AI assistant. The user asked: {query}
            The {routed['tool']} tool was used and it returned:
            {routed['result']}

            Please provide a clear, final synthesized answer for the user.
            """
            final_answer = llm.predict(synthesis_prompt)

        # Save in chat history
        st.session_state.chat_history.append({
            "user": query,
            "assistant": final_answer,
            "tool": routed["tool"],
            "context": routed["result"]
        })

        st.success("✅ Final Answer:")
        st.write(final_answer)
        st.info(f"🔧 Tool Used: {routed['tool']}")

# -------------------------
# 6. SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Chat Controls")

# 🧹 Clear Chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")

# ⬇️ Download Chat
if st.session_state.chat_history:
    chat_json = json.dumps(st.session_state.chat_history, indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        label="⬇️ Download Chat",
        data=chat_json,
        file_name="chat_history.json",
        mime="application/json"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Developed by Bushra**")

