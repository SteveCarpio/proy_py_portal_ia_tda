# ==========================================
# 🧠 LocalDocChat - RAG con Memoria y Ollama
# ==========================================
# Compatible con LangChain 1.0+, Streamlit y Ollama
# Autor: ChatGPT (versión optimizada y documentada)
# ------------------------------------------

import streamlit as st
import os

# --- LangChain imports (estructura 2025) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- Configuración del modelo y directorios ---
OLLAMA_MODEL = "llama3"   # Cambia por el modelo Ollama que tengas (mistral, phi3, etc.)
CHROMA_PATH = "chroma_db" # Carpeta donde se guardará la base vectorial

# ------------------------------------------
# 🔧 Función: Procesar e indexar un PDF
# ------------------------------------------
def process_and_index_pdf(file_path):
    """
    Carga un PDF, lo divide en fragmentos (chunks),
    genera embeddings con Ollama y los guarda en Chroma.
    """

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.success(f"✅ Documento cargado ({len(documents)} páginas).")

    # División del texto en fragmentos para mejorar la búsqueda semántica
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # tamaño de fragmento
        chunk_overlap=200,    # solapamiento entre fragmentos
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"📑 Documento dividido en {len(chunks)} fragmentos.")

    # Crear embeddings con modelo de Ollama
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Crear base vectorial Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    #vectorstore.persist()
    st.success("📚 Documento indexado y guardado en Chroma.")
    return vectorstore

# ------------------------------------------
# 🧩 Función: Crear la cadena RAG con memoria
# ------------------------------------------
def create_rag_chain(vectorstore):
    """
    Crea una función que toma una pregunta, recupera contexto relevante
    del PDF y genera una respuesta usando Ollama.
    Incluye memoria conversacional basada en el historial de chat.
    """

    retriever = vectorstore.as_retriever()
    llm = Ollama(model=OLLAMA_MODEL)

    # Prompt que guía al modelo
    prompt_template = ChatPromptTemplate.from_template(
        """
        Eres un asistente que responde basándote solo en el contexto dado.
        Si la información no está en el contexto, di claramente:
        "No encontré esa información en el documento."

        Ten en cuenta la conversación previa si ayuda a entender la pregunta.

        ----
        Historial del chat:
        {history}

        Contexto relevante del documento:
        {context}

        Pregunta actual:
        {question}
        """
    )

    def rag_query(question, history):
        """Ejecuta una consulta RAG teniendo en cuenta el historial."""
        # Recupera documentos relevantes
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prepara el historial como texto legible
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-5:]])  # últimas 5 interacciones

        # Rellena el prompt
        prompt = prompt_template.format(context=context, question=question, history=history_text)

        # Genera la respuesta con el LLM
        response = llm.invoke(prompt)
        return str(response)

    return rag_query

# ------------------------------------------
# 💬 Interfaz principal Streamlit
# ------------------------------------------
st.set_page_config(page_title="LocalDocChat - RAG con Memoria", layout="wide")
st.title("💬 LocalDocChat (RAG con Ollama y Streamlit)")
st.caption("Tu asistente local para consultar documentos PDF. Compatible con Ollama y Chroma.")

# --- Subida del archivo PDF ---
uploaded_file = st.sidebar.file_uploader("📄 Sube un archivo PDF para analizar", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Guardar PDF en carpeta local
    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"✅ Archivo guardado: {uploaded_file.name}")

    with st.spinner("🧠 Procesando e indexando el documento..."):
        vector_store = process_and_index_pdf(file_path)
        # Guardar en sesión para mantener persistencia
        st.session_state['vector_store'] = vector_store
        st.session_state['chat_history'] = []
        
        #st.experimental_rerun()


# --- Zona de Chat ---
if 'vector_store' in st.session_state:
    st.header("💭 Chatea con tu Documento")

    # Inicializar memoria si no existe
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    rag_chain = create_rag_chain(st.session_state['vector_store'])

    # Mostrar historial del chat
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
        # Añadir pregunta al historial
        st.session_state['chat_history'].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧩 Buscando en el documento y generando respuesta..."):
                response = rag_chain(prompt, st.session_state['chat_history'])
                st.markdown(response)

            # Guardar respuesta en el historial
            st.session_state['chat_history'].append({"role": "assistant", "content": response})
else:
    st.info("👈 Sube un archivo PDF en la barra lateral para comenzar.")

# --- Información en la barra lateral ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")
st.sidebar.markdown(f"📂 Base de datos Chroma: `{CHROMA_PATH}`")
st.sidebar.markdown("💡 Consejo: Usa preguntas específicas para obtener mejores respuestas.")
