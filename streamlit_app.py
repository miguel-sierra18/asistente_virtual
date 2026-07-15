from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from app import build_agent, PROJECT_ROOT, DEFAULT_DOCUMENT_DIR


st.set_page_config(page_title="Asistente NovaShop", page_icon="🛒", layout="wide")

st.title("🛒 Asistente Virtual — NovaShop")
st.caption("Consulta políticas de privacidad, devoluciones, envíos, términos y condiciones, y FAQ (RAG) o preguntas externas (Web) en un solo chat.")


def _secrets_file_exists() -> bool:
    """Verifica si existe un secrets.toml local antes de tocar st.secrets,
    para evitar el banner de aviso de Streamlit cuando solo usamos .env."""
    candidatos = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        PROJECT_ROOT / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidatos)


def sync_secrets_to_env() -> None:
    """Permite configurar las API keys vía Streamlit Secrets (útil en el deploy)."""
    if not _secrets_file_exists():
        return

    try:
        secrets_dict = dict(st.secrets)
    except Exception:
        return

    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "SERPAPI_API_KEY", "DOCUMENT_DIR"]:
        value = secrets_dict.get(key)
        if value and not os.getenv(key):
            os.environ[key] = str(value)

    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


sync_secrets_to_env()


def api_key_ref() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return "no-configurada"
    if len(key) <= 10:
        return "configurada"
    return f"{key[:4]}...{key[-6:]}"


def _resolve_document_dir() -> Path:
    configured = os.getenv("DOCUMENT_DIR")
    doc_dir = Path(configured) if configured else DEFAULT_DOCUMENT_DIR
    if not doc_dir.is_absolute():
        doc_dir = (PROJECT_ROOT / doc_dir).resolve()
    return doc_dir


@st.cache_resource
def get_agent():
    return build_agent(rebuild_index=False)


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.agent_blocked = False


# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    st.write("Claves API leídas desde `.env` (local) o `st.secrets` (Streamlit Cloud).")

    if st.button("🔄 Reconstruir índice FAISS"):
        with st.spinner("Reconstruyendo índice, esto puede tardar..."):
            try:
                build_agent(rebuild_index=True)
                st.cache_resource.clear()
                st.success("Índice reconstruido con éxito.")
            except Exception as exc:
                st.error(f"❌ Error al reconstruir el índice: {exc}")

    if st.button("🧹 Limpiar chat"):
        reset_chat()
        st.rerun()

    modo_debug = st.checkbox(
        "🔍 Modo debug (mostrar contexto usado)",
        help="Muestra el contexto crudo (RAG o Web) que el agente usó para generar cada respuesta.",
    )

    st.divider()
    st.header("📂 Colección de documentos")

    doc_dir = _resolve_document_dir()
    doc_dir.mkdir(parents=True, exist_ok=True)

    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Sube uno o varios documentos (se agregan a la colección)",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="Límite: 200 MB por archivo • PDF o CSV",
        )
        submitted = st.form_submit_button("💾 Guardar y reconstruir índice")
        if submitted and uploaded_files:
            archivos_nuevos = False
            for uploaded_file in uploaded_files:
                destino = doc_dir / uploaded_file.name
                if destino.exists():
                    st.warning(f"⚠️ '{uploaded_file.name}' ya existe y no fue sustituido.")
                    continue
                with open(destino, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                archivos_nuevos = True

            if archivos_nuevos:
                with st.spinner("Reconstruyendo índice con los nuevos documentos..."):
                    try:
                        build_agent(rebuild_index=True)
                        st.cache_resource.clear()
                        st.success("Documentos agregados y índice reconstruido.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ Error al procesar los documentos: {exc}")

    st.subheader("📑 Documentos actuales")
    archivos_actuales = sorted(
        [f for f in doc_dir.glob("*") if f.suffix.lower() in (".csv", ".pdf")]
    )
    if not archivos_actuales:
        st.write("No hay documentos en la colección todavía.")
    else:
        for archivo in archivos_actuales:
            col1, col2 = st.columns([0.8, 0.2])
            col1.write(f"{'📄' if archivo.suffix.lower() == '.pdf' else '📊'} {archivo.name}")
            if col2.button("🗑️", key=f"del_{archivo.name}"):
                archivo.unlink()
                with st.spinner("Reconstruyendo índice..."):
                    try:
                        build_agent(rebuild_index=True)
                    except Exception:
                        pass  # si no quedan documentos, build_agent puede fallar
                st.cache_resource.clear()
                st.rerun()

    st.divider()
    st.caption(
        "💡 Ejemplos de preguntas:\n\n"
        "- ¿Cuántos días tengo para devolver un producto?\n"
        "- ¿Qué métodos de pago aceptan?\n"
        "- ¿Cuánto cuesta el envío estándar?\n"
        "- ¿Cuál es el tipo de cambio del dólar hoy? (usa Web)"
    )


# --- MAIN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_blocked" not in st.session_state:
    st.session_state.agent_blocked = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pregunta = st.chat_input("✍️ Escribe tu pregunta sobre NovaShop...")

if pregunta:
    if st.session_state.agent_blocked:
        st.warning("⚠️ El agente quedó bloqueado por un error previo. Pulsa 'Limpiar chat' para reintentar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            try:
                agente = get_agent()
                resultado = agente.invoke(
                    {
                        "pregunta": pregunta,
                        "fuente": "",
                        "contexto": "",
                        "respuesta": "",
                    }
                )

                fuente = resultado.get("fuente", "N/D")
                respuesta = resultado.get("respuesta", "No se pudo generar respuesta.")

                st.markdown(f"**Fuente usada:** {fuente}")
                st.markdown(respuesta)

                if modo_debug:
                    with st.expander("🔍 Contexto usado (debug)"):
                        st.text(resultado.get("contexto", "(vacío)"))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"**Fuente usada:** {fuente}\n\n{respuesta}",
                    }
                )
            except Exception as exc:
                error_msg = f"❌ Error ejecutando el agente: {exc}\n\nReferencia API key: {api_key_ref()}"
                st.error(error_msg)
                st.session_state.agent_blocked = True
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )