# ==========================================
# 🧠 LocalDocChat - Mejora para PDFs grandes, RAG con Ollama local
# ==========================================
# Autor: Copilot (mejoras sugeridas)
# Notas: He introducido:
#  - index registry para evitar re-indexar ficheros iguales
#  - indexación incremental / persistente por fichero (checksum)
#  - batching/persistencia explícita en Chroma
#  - retriever MMR + control del tamaño del contexto enviado al LLM
#  - metadata por página (origen / página)
#  - ajustes de chunking recomendados para documentos largos
#  - progresos en Streamlit y manejo más robusto de estado
# Compatibilidad: adaptado para langchain 1.x con wrappers de Ollama
# ==========================================


import os
import uuid
import json
import hashlib
from datetime import datetime
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# -----------------------
# Configuración (ajusta según tu entorno)
# -----------------------
PROJECT_PATH = "/home/robot/Python/proy_py_portal_ia_tda"
DATA_PATH = os.path.join(PROJECT_PATH, "data")
CHROMA_BASE_PATH = os.path.join(PROJECT_PATH, "chroma_db")
INDEX_REGISTRY_PATH = os.path.join(CHROMA_BASE_PATH, "index_registry.json")
OLLAMA_MODEL = "tda-llama3"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_BASE_PATH, exist_ok=True)

# -----------------------
# Utilidades
# -----------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_index_registry():
    if os.path.exists(INDEX_REGISTRY_PATH):
        try:
            with open(INDEX_REGISTRY_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}

def save_index_registry(reg):
    with open(INDEX_REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, ensure_ascii=False, indent=2)

def new_chroma_path_for_file(filename, checksum):
    # Use a deterministic directory per file-checksum so we can reuse index later
    safe_name = Path(filename).stem
    session_id = f"{safe_name}_{checksum[:8]}"
    session_path = os.path.join(CHROMA_BASE_PATH, session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_path

# -----------------------
# Indexación optimizada para PDFs grandes
# -----------------------
def process_and_index_pdf(file_path, optional_progress_callback=None,
                          chunk_size=1200, chunk_overlap=200, max_chars_context=3500):
    """
    - Reusa índices si el fichero ya fue indexado (por checksum).
    - Guarda metadata (filename, page).
    - Persiste la base de vectores en CHROMA_BASE_PATH/<file>_<checksum>
    - Devuelve el vectorstore y la ruta donde está persistido.
    """
    checksum = file_sha256(file_path)
    registry = load_index_registry()
    if checksum in registry:
        chroma_path = registry[checksum]["chroma_path"]
        st.info(f"🔁 Índice existente encontrado para este fichero. Reusando: {chroma_path}")
        # Abrir vectorstore persistente
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        return vectorstore, chroma_path

    # No existe índice: crear uno nuevo
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    st.success(f"✅ Documento cargado ({len(pages)} páginas).")

    # Ajusta chunk_size / overlap para balance entre contexto y número de embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(pages)
    #st.info(f"📑 Documento dividido en {len(chunks)} fragmentos (chunks).")

    # Añadir metadata por chunk (fuente y página si está disponible)
    for c in chunks:
        # PyPDFLoader normalmente añade metadata de página en 'metadata' o 'page'
        # Conservamos filename y checksum para trazabilidad
        md = c.metadata if hasattr(c, "metadata") and isinstance(c.metadata, dict) else {}
        md["source_file"] = os.path.basename(file_path)
        # si loader incluyó number_of_page u 'page' lo guardamos, si no, lo dejamos
        c.metadata = md

    # Crear / configurar embeddings y Chroma
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    chroma_path = new_chroma_path_for_file(os.path.basename(file_path), checksum)
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # Añadir documentos en batch y persistir (esto evita usar from_documents que embeddea todo a la vez sin control)
    # Si Chroma local soporta add_documents, lo usamos.
    #st.info("🔎 Generando embeddings y almacenando en Chroma (esto puede tardar en PDFs grandes)...")
    progress = optional_progress_callback or (lambda i, n: None)
    total = len(chunks)
    batch_size = 64  # ajustable: cuantos chunks mandar por batch (depende de memoria / rendimiento)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        progress(min(i + batch_size, total), total)

    #vectorstore.persist()

    # Después de add_documents(...) en tu loop de batches
    # Intentar persistir de forma segura según la implementación disponible
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    else:
        # probar con client o _client de chromadb si existe
        client = getattr(vectorstore, "client", None) or getattr(vectorstore, "_client", None)
        try:
            if client and hasattr(client, "persist"):
                client.persist()
            else:
                # Si no hay método explícito, informar (pero en muchas implementaciones
                # la persistencia ya ocurre si creaste la store con persist_directory)
                st.info("Info: el objeto Chroma no expone persist(); si usaste persist_directory, los datos pueden haberse guardado automáticamente.")
        except Exception as e:
            st.warning(f"No pude invocar persist en el cliente Chroma: {e}")



    # Registrar en index registry para reuso futuro
    registry[checksum] = {
        "chroma_path": chroma_path,
        "file": os.path.basename(file_path),
        "indexed_at": datetime.utcnow().isoformat() + "Z",
        "chunks": total,
    }
    save_index_registry(registry)
    st.success(f"📚 Documento indexado y persistido en Chroma: {chroma_path} ({total} chunks)")
    return vectorstore, chroma_path

# -----------------------
# RAG query: recuperación + llamada al LLM
# -----------------------
def create_rag_query_fn(vectorstore, max_context_chars=3500, k=6, fetch_k=12):
    # Use MMR para diversidad en documentos (mejor para preguntas amplias)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})
    llm = Ollama(model=OLLAMA_MODEL, temperature=0)

    system_instructions = (
        "Eres un asistente que responde SOLO con la información que aparece en el contexto.\n"
        "Si la respuesta no está en el contexto, responde exactamente: "
        "\"No encontré esa información en el documento.\"\n"
        "Responde en el mismo idioma de la pregunta."
    )

    def rag_query(question, history):
        # Obtener documentos relevantes
        docs = retriever.get_relevant_documents(question)
        # Construir contexto limitado por caracteres para evitar 'context overflow'
        context_chunks = []
        chars = 0
        for d in docs:
            piece = d.page_content.strip()
            meta = d.metadata or {}
            header = ""
            # adjuntar referencia de página si existe
            if meta.get("source_file"):
                header = f"[{meta.get('source_file')}{( ' | p.' + str(meta.get('page')) ) if meta.get('page') else ''}] "
            combined = header + piece
            if chars + len(combined) > max_context_chars and len(context_chunks) > 0:
                break
            context_chunks.append(combined)
            chars += len(combined)
        context = "\n\n---\n\n".join(context_chunks)

        # Historial limitado (últimas X interacciones)
        history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-6:]]) if history else ""

        prompt = (
            f"{system_instructions}\n\n"
            f"Historial del chat:\n{history_text}\n\n"
            f"Contexto relevante:\n{context}\n\n"
            f"Pregunta actual:\n{question}\n\n"
            "Responde de forma clara y cita (entre corchetes) la fuente/página cuando sea posible."
        )

        # Llamada al LLM (se usa invoke para compatibilidad con el wrapper Ollama)
        try:
            response = llm.invoke(prompt)
            # Algunos wrappers devuelven objetos; convertir a str si es necesario
            return str(response).strip()
        except Exception as e:
            return f"Error llamando al LLM: {e}"

    return rag_query

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="LocalDocChat Mejorado", layout="wide")
st.title("💬 LocalDocChat — (RAG + Ollama)")
st.caption("Optimizado para grandes PDFs: reuso de índices, control del contexto y MMR.")

# Sidebar: reset / info
#st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Limpiar / Resetear chat"):
    st.session_state.clear()
    st.success("✅ Estado reseteado. Vuelve a subir el PDF.")

#st.sidebar.markdown("---")
#st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")
#st.sidebar.markdown(f"📂 Carpeta PDFs: `{DATA_PATH}`")
#st.sidebar.markdown(f"📂 Carpeta índices Chroma: `{CHROMA_BASE_PATH}`")

# Subida de archivo
uploaded = st.sidebar.file_uploader("📄 Sube un PDF (puede ser grande)", type="pdf")
if uploaded is not None:
    os.makedirs(DATA_PATH, exist_ok=True)
    save_path = os.path.join(DATA_PATH, uploaded.name)
    with open(save_path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    #st.sidebar.success(f"✅ Guardado: {uploaded.name}")

    # Inicializar progreso visual
    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()

    def progress_cb(done, total):
        frac = int(done / total * 100) if total else 100
        progress_bar.progress(frac)
        progress_text.text(f"Indexando: {done}/{total} chunks")

    with st.spinner("🧠 Procesando e indexando (esto puede tardar dependiendo del tamaño)..."):
        vectorstore, chroma_path = process_and_index_pdf(save_path, optional_progress_callback=progress_cb)
        st.session_state["vectorstore_path"] = chroma_path
        st.session_state["last_indexed_file"] = save_path

    progress_bar.empty()
    #progress_text.text("Indexación completada.")

# Si ya hay un vectorstore persistente en sesión o registrado, cargarlo
if "vectorstore_path" in st.session_state and os.path.exists(st.session_state["vectorstore_path"]):
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectorstore = Chroma(persist_directory=st.session_state["vectorstore_path"], embedding_function=embeddings)
    st.session_state["vector_store_obj"] = vectorstore

# Chat UI
if "vector_store_obj" in st.session_state:
    st.header("💭 Chatea con tu documento (índice persistente)")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    rag_query = create_rag_query_fn(st.session_state["vector_store_obj"])

    # Mostrar historial
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Haz una pregunta sobre el documento..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🧩 Buscando y generando respuesta..."):
                resp = rag_query(prompt, st.session_state["chat_history"])
                st.markdown(resp)
            st.session_state["chat_history"].append({"role": "assistant", "content": resp})

else:
    st.info("Sube un PDF en la barra lateral para empezar a indexarlo y chatear con él.")

# footer: mostrar ruta de chroma si existe
if "vectorstore_path" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"📂 Índice activo: `{st.session_state['vectorstore_path']}`")
    st.sidebar.markdown(f"🧩 Modelo LLM: **{OLLAMA_MODEL}**")