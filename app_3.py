# ==========================================
# 🧠 LocalDocChat - RAG con Memoria y Ollama
# ==========================================
# Autor: ChatGPT
# Descripción: Chat con PDFs usando RAG + Ollama + Streamlit
# Compatible con LangChain 1.0.x, Chroma 1.0.x y Streamlit moderno
# ==========================================

import streamlit as st
import os

# --- LangChain imports (estructura moderna) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- Configuración ---
OLLAMA_MODEL = "llama3"        # Cambia al modelo que tengas en Ollama
CHROMA_PATH = "chroma_db"      # Carpeta para la base vectorial persistente

# ==========================================
# 🔧 Función: Procesar e indexar PDF
# ==========================================
def process_and_index_pdf(file_path):
    """
    Carga un PDF, lo divide en fragmentos (chunks),
    genera embeddings y los indexa en Chroma.
    Chroma guarda automáticamente los vectores si se usa persist_directory.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.success(f"✅ Documento cargado ({len(documents)} páginas).")

    # División en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"📑 Documento dividido en {len(chunks)} fragmentos.")

    # Embeddings con Ollama
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Indexación en Chroma (persistente)
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
        # Solo usamos las últimas 5 interacciones para no saturar el prompt
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])
        prompt = prompt_template.format(context=context, question=question, history=history_text)
        response = llm.invoke(prompt)
        return str(response)

    return rag_query

# ==========================================
# 💬 Interfaz Streamlit
# ==========================================
st.set_page_config(page_title="LocalDocChat - RAG con Memoria", layout="wide")
st.title("💬 LocalDocChat (RAG con Ollama y Streamlit)")
st.caption("Consulta tus PDFs de manera inteligente. Memoria y RAG incluidos.")

# --- Subida de PDF ---
uploaded_file = st.sidebar.file_uploader("📄 Sube un archivo PDF para analizar", type="pdf")

if uploaded_file is not None:
    # Guardar PDF localmente
    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ Archivo guardado: {uploaded_file.name}")

    # Procesar PDF y guardar vectorstore en sesión
    with st.spinner("🧠 Procesando e indexando el documento..."):
        st.session_state['vector_store'] = process_and_index_pdf(file_path)
        st.session_state['chat_history'] = []

# --- Zona de chat ---
if 'vector_store' in st.session_state:
    st.header("💭 Chatea con tu Documento")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    rag_chain = create_rag_chain(st.session_state['vector_store'])

    # Mostrar historial
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

# --- Barra lateral ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")
st.sidebar.markdown(f"📂 Base de datos Chroma: `{CHROMA_PATH}`")
st.sidebar.markdown("💡 Consejo: Usa preguntas concretas para mejores respuestas.")
