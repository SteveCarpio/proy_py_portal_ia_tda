# Módulos de Streamlit y OS
import streamlit as st
import os

# Document loaders, text splitters, y embeddings del paquete langchain-community
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Base de datos vectorial de Chroma del paquete langchain-chroma
from langchain_chroma import Chroma

# LLM de Ollama del paquete langchain-community
from langchain_community.llms.ollama import Ollama

# Componentes de la cadena RAG
#from langchain.chains.retrieval import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Configuración ---
# Puedes cambiar 'llama2' por el nombre del modelo que tengas en Ollama (ej: 'mistral', 'llama3')
OLLAMA_MODEL = "tda-llama3" 
CHROMA_PATH = "chroma_db" 

# --- Funciones RAG ---

def process_and_index_pdf(file_path):
    """Carga el PDF, lo divide, genera embeddings y lo indexa en Chroma."""
    
    # 1. Carga del Documento
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.success(f"Documento cargado. Total de páginas: {len(documents)}")

    # 2. División del Texto (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"Texto dividido en {len(chunks)} fragmentos (chunks).")

    # 3. Creación de Embeddings
    # Usamos un modelo de embeddings de Ollama (generalmente 'llama2' o un modelo específico si lo configuras)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # 4. Indexación en Base de Datos Vectorial (Chroma)
    # Se sobrescribe la base de datos para el nuevo archivo
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    st.success("Documento indexado con éxito y listo para consultar.")
    return vectorstore

def create_rag_chain(vectorstore):
    """Crea la cadena de recuperación y generación para la consulta RAG."""
    
    # Template del prompt: Define cómo debe responder el LLM
    prompt_template = ChatPromptTemplate.from_template(
        """
        Responde a la pregunta basándote únicamente en el contexto proporcionado.
        Si no puedes encontrar la respuesta en el contexto, simplemente di que la información no está disponible en el documento.

        Contexto: {context}
        Pregunta: {input}
        """
    )

    # Inicializa el LLM de Ollama
    llm = Ollama(model=OLLAMA_MODEL)
    
    # Crea la cadena que combina el contexto (stuff documents)
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Crea el recuperador (retriever) a partir de la base de datos Chroma
    retriever = vectorstore.as_retriever()
    
    # Combina el recuperador y la cadena de documentos para formar la cadena RAG completa
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

def run_query(rag_chain, query):
    """Ejecuta la consulta RAG y retorna la respuesta."""
    response = rag_chain.invoke({"input": query})
    return response['answer']

# --- Interfaz de Streamlit ---

st.set_page_config(page_title="LocalDocChat - NotebookLM Self-Hosted", layout="wide")
st.title("LocalDocChat 💬 (RAG con Ollama y Streamlit)")
st.caption("Una alternativa self-hosted a NotebookLM. Desarrollado por un Gemini 🤖.")

# 1. Subida de Archivo y Procesamiento
uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo PDF para indexar", 
    type="pdf", 
    key="pdf_uploader"
)

if uploaded_file is not None:
    # Guardar el archivo subido en la carpeta data/
    if not os.path.exists("data"):
        os.makedirs("data")
        
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"Archivo guardado: {uploaded_file.name}")
    
    # Procesar e indexar el archivo
    with st.spinner("Procesando e indexando el documento (Chunking y Embeddings)... Esto puede tardar unos minutos."):
        vector_store = process_and_index_pdf(file_path)
        # Almacenar la vectorstore en la sesión para usarla en el chat
        st.session_state['vector_store'] = vector_store
        st.session_state['chat_history'] = []
        st.experimental_rerun() # Recargar para limpiar el uploader y mostrar el chat

# 2. Área de Chat
if 'vector_store' in st.session_state:
    st.header("Chatea con tu Documento Indexado")
    
    # Inicializar historial de chat
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Crear la cadena RAG
    rag_chain = create_rag_chain(st.session_state['vector_store'])

    # Mostrar historial de chat
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando y generando respuesta..."):
                # Ejecutar la consulta RAG
                response_text = run_query(rag_chain, prompt)
                st.markdown(response_text)
                
            st.session_state['chat_history'].append({"role": "assistant", "content": response_text})
else:
    st.info("Sube un archivo PDF en la barra lateral izquierda para comenzar.")
    
st.sidebar.markdown("---")
st.sidebar.markdown(f"Modelo LLM Usado: **{OLLAMA_MODEL}**")
st.sidebar.markdown(f"Directorio de Chroma DB: `{CHROMA_PATH}`")