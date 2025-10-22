# ==========================================
# 🧠 LocalDocChat - RAG con Memoria y Ollama Local
# ==========================================
# Autor: ChatGPT
# Descripción: Chat con PDFs usando RAG + Ollama + Streamlit
# Compatible con:
#   langchain==1.0.2
#   langchain-community==0.4
#   langchain-chroma==1.0.0
#   streamlit>=1.38
# ==========================================

import streamlit as st
import os
import uuid
from datetime import datetime
from pathlib import Path

# --- LangChain imports para Ollama local ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- Configuración ---
PROJECT_PATH = "/home/robot/Python/proy_py_portal_ia_tda"
DATA_PATH = os.path.join(PROJECT_PATH, "data")  # PDFs
os.makedirs(DATA_PATH, exist_ok=True)

OLLAMA_MODEL = "tda-llama3"  # Modelo Ollama local

# Carpeta base para Chroma; cada sesión crea subcarpeta única
CHROMA_BASE_PATH = os.path.join(PROJECT_PATH, "chroma_db")
os.makedirs(CHROMA_BASE_PATH, exist_ok=True)

# ==========================================
# 🔧 Función: Crear carpeta única de Chroma para la sesión
# ==========================================
def new_chroma_session_path():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    session_path = os.path.join(CHROMA_BASE_PATH, session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_path

# ==========================================
# 🔧 Función: Procesar e indexar PDF
# ==========================================
def process_and_index_pdf(file_path, chroma_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.success(f"✅ Documento cargado ({len(documents)} páginas).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # tamaño de fragmento
        chunk_overlap=200,      # solapamiento entre fragmentos
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"📑 Documento dividido en {len(chunks)} fragmentos.")

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    st.success(f"📚 Documento indexado en Chroma: {chroma_path}")
    return vectorstore

# ==========================================
# 🧩 Función: Crear cadena RAG con memoria
# ==========================================
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Eres un asistente que responde solo con información disponible en el contexto.
        Si la información no está en el contexto, di:
        "No encontré esa información en el documento."

        Historial del chat:
        {history}

        Contexto relevante:
        {context}

        Pregunta actual:
        {question}
        """
    )

    def rag_query(question, history):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])
        prompt = prompt_template.format(context=context, question=question, history=history_text)
        response = llm.invoke(prompt)
        return str(response)

    return rag_query

# ==========================================
# 💬 Interfaz Streamlit
# ==========================================
st.set_page_config(page_title="LocalDocChat Safe", layout="wide")
st.title("💬 DocChat (RAG con Ollama local seguro)")
st.caption("Consulta tus PDFs de manera inteligente. Memoria y RAG incluidos.")

# --- Botón limpiar historial ---
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Limpiar / Resetear chat"):
    st.session_state['chat_history'] = []
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']
    st.success("✅ Chat reseteado. Sube un PDF para nueva sesión.")

# --- Subida de PDF ---
uploaded_file = st.sidebar.file_uploader("📄 Sube un PDF", type="pdf")

if uploaded_file is not None:
    # Guardar PDF
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Archivo guardado: {uploaded_file.name}")

    # Resetear chat y vectorstore
    st.session_state['chat_history'] = []
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']

    # Crear carpeta Chroma única para esta sesión
    chroma_path = new_chroma_session_path()
    st.session_state['chroma_path'] = chroma_path

    with st.spinner("🧠 Procesando e indexando PDF..."):
        st.session_state['vector_store'] = process_and_index_pdf(file_path, chroma_path)

# --- Chat ---
if 'vector_store' in st.session_state:
    st.header("💭 Chatea con tu Documento")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    rag_chain = create_rag_chain(st.session_state['vector_store'])

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🧩 Buscando y generando respuesta..."):
                response = rag_chain(prompt, st.session_state['chat_history'])
                st.markdown(response)
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

# --- Barra lateral info ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")
if 'chroma_path' in st.session_state:
    st.sidebar.markdown(f"📂 Carpeta Chroma sesión: `{st.session_state['chroma_path']}`")
st.sidebar.markdown(f"📂 Carpeta PDFs: `{DATA_PATH}`")
