import os
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from llama_cpp import Llama

from byob_ingest_final import process_pdfs_and_create_vector_store

# --------------------
# 1) Initialize LLM & Reranker
# --------------------
llm = Llama(
    model_path=os.getenv(
        "LLAMA_MODEL_PATH",
        "/app/models/llama/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
    ),
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=27,
)

ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir=os.getenv(
        "MSMARCO_MODEL_DIR",
        "/app/reranker/ms-marco-MiniLM-L-12-v2"
    )
)
EMBED_MODEL = os.getenv(
    "BGE_EMBED_DIR",
    "/app/embeddings/bge-large-en-v1.5"
)

# --------------------
# 2) Streamlit Setup
# --------------------
st.set_page_config(page_title="BYOB App")
st.title("Build Your Own Bot Demo Application")

# Initialize session state
if "bot_created" not in st.session_state:
    st.session_state.bot_created = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
bot_name        = st.sidebar.text_input("Enter Bot's Name")
uploaded_pdfs   = st.sidebar.file_uploader("Enter PDFs", type="pdf", accept_multiple_files=True)


default_prompt = "Your role is to provide accurate and helpful answers to user queries based solely on the context retrieved from the provided documents."


prompt_template = st.sidebar.text_area("Enter Prompt", value=default_prompt, key='prompt', height=100)
create_bot      = st.sidebar.button("Create Bot")

# --------------------
# 3) Bot Creation: PDF → JSON + FAISS
# --------------------
if create_bot:
    if bot_name and uploaded_pdfs and prompt_template:
        with st.spinner("Processing PDFs and building vector store..."):
            out = Path("output")
            out.mkdir(exist_ok=True)
            for pdf in uploaded_pdfs:
                dest = out / pdf.name
                with open(dest, "wb") as f:
                    f.write(pdf.getbuffer())
                process_pdfs_and_create_vector_store(str(dest))

            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            st.session_state.vectorstore = FAISS.load_local(
                "output/vector_store_chunks",
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.session_state.bot_created = True
        st.success("Bot created successfully.")
    else:
        st.error("Please complete all fields before creating the bot.")

# --------------------
# 4) Retrieval & Rerank
# --------------------
def retrieve_and_rerank(query: str):
    vs = st.session_state.vectorstore
    if vs is None:
        return "Bot not initialized. Please upload PDFs and click Create Bot.", ""

    # FAISS retrieval
    docs = vs.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(query)
    if not docs:
        return "No relevant context found.", ""

    passages = [{"text": d.page_content} for d in docs if d.page_content.strip()]
    if not passages:
        return "Retrieved documents contained no valid text.", ""

    # FlashRank reranking
    try:
        req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(req)
    except Exception:
        # fallback to first 3 FAISS docs
        fb = "\n".join(d.page_content for d in docs[:3])
        return fb, fb

    if not results or any("text" not in r for r in results):
        fb = "\n".join(d.page_content for d in docs[:3])
        return fb, fb

    top = results[:3]
    ctx_snip = "\n".join(r["text"] for r in top)
    return ctx_snip, " ".join(r["text"] for r in top)

# --------------------
# 5) Prompt & LLM (synchronous, no streaming)
# --------------------
def make_prompt(bot_name: str, template: str, context: str, question: str) -> str:
    return f"""
You are an intelligent chat AI assistant. Your role is to provide accurate and helpful answers to user queries based SOLELY on the context retrieved from the provided documents.
    
    Instructions:
			Greeting Response: If the input is a greeting (e.g., "hi," "hey," "hello," "whatsup," "hi I am {bot_name}," "hey this is {bot_name}"), respond with an appropriate greeting and introduce yourself as {bot_name}, an intelligent chat assisant.
			Other Queries: If the input is not a greeting, proceed with answering the query as usual.
			Relevance: The query should be STRICTLY related to retrieved context, otherwise reply with "This query is not relevant to uploaded documents. Please ask questions related to the uploaded documents."
			Name Distinction: Always distinguish between different individuals, even if their names sound similar.
			Context Selection: Choose the appropriate context(s) that can answer the question. Use these context(s) to draft the response. If the context does not contain relevant information for the asked question, respond with "I am sorry, I am not able to provide an answer for this question." and do not add any other information.
			
    Response Format for Context-Based Questions:
            
        1. **Strict Context Adherence**: {template}
        2. **Clarity and Conciseness**: Provide clear and concise answers. Avoid unnecessary elaboration.
        3. **Professional Tone**: Maintain a professional and polite tone in all responses.
        4. **Reference Context**: When applicable, refer to the specific section of the document where the information was found to support your answer.
        5. **GUIDELINES**:
            Do not use any outside knowledge, or make assumptions beyond what is presented in the context.
            DO NOT under any circumstances modify any URLs or hyperlinks in the context.  
            Avoid mentioning unrelated or random data. 
            Avoid speculative or conjectural information that does not directly relate to retrieved context confirmed expertise and offerings.			

   "Question": {question}

   "Context": {context}

"""

def generate_answer(question: str, context: str) -> str:
    prompt = make_prompt(bot_name, prompt_template, context, question)
    # Synchronous call
    result = llm(
        prompt,
        max_tokens=512,
        temperature=0.3,
        top_p=0.75,
        top_k = 50,
        stream=False
    )
    # The llama_cpp call returns a dict with 'choices' -----> I've to modify this part for streaming tokens!!
    text = result.get("choices", [{}])[0].get("text", "").strip()
    return text or "Sorry, I couldn’t generate an answer."

#Streaming is accurately done in byob_final_test

# --------------------
# 6) Chat Interface
# --------------------
user_query = st.chat_input("Ask your question about the uploaded document...")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query:
    # Save user
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        if not st.session_state.bot_created:
            resp = "Please create the bot first by uploading PDFs & clicking Create Bot."
        else:
            ctx, _ = retrieve_and_rerank(user_query)
            if "no relevant context" in ctx.lower():
                resp = ctx
            else:
                resp = generate_answer(user_query, ctx)

        st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

