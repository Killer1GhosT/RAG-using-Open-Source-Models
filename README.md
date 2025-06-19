Prerequisites
1) Create Folder "embeddings" and download BAAI/bge-large-en-v1.5 from huggingFace {model should be placed inside embeddings}
2) Create Folder "models/llama" and download  Meta LLaMA 3 8B Instruct (Q8_0.gguf) from LMStudio {model should be placed inside models/llama/(your llama model)}
3) Alternative of download is to pull the models directly using HuggingFace API

How it works?

Frontend: Built with Streamlit for a clean, interactive interface.

Embedding & Retrieval: Uses BAAI/bge-large-en-v1.5 to embed PDF chunks and FAISS for fast vector search.

Keyword Metadata: stored in a JSON file and Chromadb for Chroma+FAISS retrieval.

Reranker: Improves answer quality using FlashRank (ms-marco-MiniLM-L-12-v2) to re-rank the most relevant chunks.

LLM: Uses Meta LLaMA 3 8B Instruct (Q8_0.gguf) via llama-cpp-python for final response generation — fully local. 

Context Fusion: FAISS retrieval, reranks the results, and feeds the best context + Prompt (in-code) + User Prompt to the LLM.
