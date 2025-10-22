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
import shutil

# --- LangChain imports para Ollama local ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- Configuración ---
OLLAMA_MODEL = "llama3"        # Cambia al modelo que tengas en Ollama local
CHROMA_PATH = os.path.expanduser("~/localdocchat/chroma_db")  # Ruta segura para Chroma
DATA_PATH = os.path.expanduser("~/localdocchat/data")          # Carpeta para PDFs

# Crear carpetas si no existen
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# ==========================================
# 🔧 Función: Procesar e indexar PDF
# ==========================================
def process_and_index_pdf(file_path):
    """
    Carga un PDF, lo divide en fragmentos (chunks),
    genera embeddings con Ollama local y los indexa en Chroma.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.success(f"✅ Documento cargado ({len(documents)} páginas).")

    # División en fragmentos para mejorar búsqueda semántica
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"📑 Documento dividido en {len(chunks)} fragmentos.")

    # Embeddings con Ollama local
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Indexación en Chroma (persistencia automática)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    st.success("📚 Documento indexado y guardado en Chroma automáticamente.")
    return vectorstore

# ==========================================
# 🧩 Función: Crear cadena RAG con memoria
# ==========================================
def create_rag_chain(vectorstore):
    """
    Devuelve una función que toma una pregunta y devuelve la respuesta usando RAG.
    Incluye historial de chat (memoria).
    """
    retriever = vectorstore.as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)

    # Prompt guía
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
        """Consulta RAG con historial de chat."""
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        # Últimas 5 interacciones del chat
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])
        prompt = prompt_template.format(context=context, question=question, history=history_text)
        response = llm.invoke(prompt)
        return str(response)

    return rag_query

# ==========================================
# 💬 Interfaz Streamlit
# ==========================================
st.set_page_config(page_title="LocalDocChat - RAG con Memoria", layout="wide")
st.title("💬 LocalDocChat (RAG con Ollama local y Streamlit)")
st.caption("Consulta tus PDFs de manera inteligente. Memoria y RAG incluidos.")

# --- Botón para limpiar historial y vectorstore ---
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Limpiar / Resetear todo"):
    st.session_state['chat_history'] = []
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']

    # Borrar carpeta antigua y crear una vacía con permisos
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    st.success("✅ Historial y base de datos reseteados. Sube un nuevo PDF.")

# --- Subida de PDF ---
uploaded_file = st.sidebar.file_uploader("📄 Sube un archivo PDF para analizar", type="pdf")

if uploaded_file is not None:
    # Guardar PDF
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Archivo guardado: {uploaded_file.name}")

    # --- Resetear chat y vectorstore al subir PDF nuevo ---
    st.session_state['chat_history'] = []
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']

    # Borrar base antigua y crear carpeta con permisos
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Procesar PDF y crear nueva base
    with st.spinner("🧠 Procesando e indexando el documento..."):
        st.session_state['vector_store'] = process_and_index_pdf(file_path)

# --- Zona de chat ---
if 'vector_store' in st.session_state:
    st.header("💭 Chatea con tu Documento")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    rag_chain = create_rag_chain(st.session_state['vector_store'])

    # Mostrar historial de chat
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
        # Guardar pregunta en historial
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧩 Buscando en el documento y generando respuesta..."):
                response = rag_chain(prompt, st.session_state['chat_history'])
                st.markdown(response)
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

# --- Información barra lateral ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")
st.sidebar.markdown(f"📂 Base de datos Chroma: `{CHROMA_PATH}`")
st.sidebar.markdown("💡 Consejo: Haz preguntas concretas para mejores respuestas.")
