from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph

from src.document_loader import cargar_coleccion


class AgentState(TypedDict):
    pregunta: str
    fuente: str
    contexto: str
    respuesta: str


PROJECT_ROOT = Path(__file__).resolve().parent
# Carpeta con los documentos sobre los que el agente responde (RAG).
# Puede contener varios PDFs y/o CSVs a la vez (políticas, ventas, docs técnicos, etc.)
DEFAULT_DOCUMENT_DIR = PROJECT_ROOT / "data" / "documentos"
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"

load_dotenv()


def _extraer_contexto_serpapi(resultados: dict) -> str:
    """
    SerpAPIWrapper.run() a veces devuelve 'No good search result found.'
    porque su resumen automático solo mira un par de campos. Aquí extraemos
    directamente del JSON crudo: answer_box, knowledge_graph y los primeros
    snippets orgánicos, para no perder información que sí llegó de Google.
    """
    partes = []

    answer_box = resultados.get("answer_box")
    if isinstance(answer_box, dict):
        for campo in ("answer", "snippet", "result", "title"):
            valor = answer_box.get(campo)
            if valor:
                partes.append(str(valor))
                break

    knowledge_graph = resultados.get("knowledge_graph")
    if isinstance(knowledge_graph, dict):
        descripcion = knowledge_graph.get("description")
        if descripcion:
            partes.append(str(descripcion))

    for resultado in resultados.get("organic_results", [])[:3]:
        snippet = resultado.get("snippet")
        if snippet:
            partes.append(str(snippet))

    return "\n".join(partes)


def _get_google_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _validate_env() -> None:
    missing = []
    if not _get_google_api_key():
        missing.append("GEMINI_API_KEY/GOOGLE_API_KEY")
    if not os.getenv("SERPAPI_API_KEY"):
        missing.append("SERPAPI_API_KEY")

    if missing:
        raise RuntimeError(
            "Faltan variables de entorno: "
            + ", ".join(missing)
            + ". Crea un archivo .env basado en .env.example"
        )


def _resolve_document_dir() -> Path:
    configured = os.getenv("DOCUMENT_DIR")
    doc_dir = Path(configured) if configured else DEFAULT_DOCUMENT_DIR
    if not doc_dir.is_absolute():
        doc_dir = (PROJECT_ROOT / doc_dir).resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


def get_embeddings(google_api_key: str | None) -> GoogleGenerativeAIEmbeddings:
    """Prueba modelos de embeddings en orden y usa el primero disponible."""
    modelos_a_probar = [
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
        "embedding-001",
    ]
    errores: list[str] = []

    for nombre_modelo in modelos_a_probar:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=nombre_modelo,
                google_api_key=google_api_key,
            )
            embeddings.embed_query("test")
            if nombre_modelo != "models/gemini-embedding-001":
                print(f"Aviso: usando fallback de embeddings: {nombre_modelo}")
            return embeddings
        except Exception as e:
            errores.append(f"{nombre_modelo}: {e}")
            continue

    raise RuntimeError(
        "No se pudo inicializar ningún modelo de embeddings. "
        "Revisa tu API key/permisos. Detalles: " + " | ".join(errores)
    )


def _build_or_load_vectorstore(embeddings: GoogleGenerativeAIEmbeddings, rebuild: bool = False):
    index_faiss = INDEX_DIR / "index.faiss"
    index_pkl = INDEX_DIR / "index.pkl"

    if not rebuild and index_faiss.exists() and index_pkl.exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    doc_dir = _resolve_document_dir()
    # cargar_coleccion recorre TODOS los .csv/.pdf de la carpeta y los
    # fusiona; cada uno usa la estrategia de document_loader que le
    # corresponda (agregación automática para CSVs grandes, chunking para PDFs).
    chunks = cargar_coleccion(str(doc_dir))

    vectorstore = _embed_en_lotes_con_reintentos(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def _es_error_rate_limit(exc: Exception) -> bool:
    texto = str(exc)
    return "429" in texto or "RESOURCE_EXHAUSTED" in texto or "rate limit" in texto.lower()


def _embed_en_lotes_con_reintentos(
    chunks,
    embeddings: GoogleGenerativeAIEmbeddings,
    tamano_lote: int = 20,
    pausa_entre_lotes: float = 2.0,
    max_reintentos: int = 5,
):
    """
    Construye el índice FAISS embebiendo los documentos en lotes pequeños,
    con pausas entre lotes y reintentos con espera exponencial si la API
    de Gemini responde con 429/RESOURCE_EXHAUSTED (límite del plan gratuito).
    """
    vectorstore = None
    total_lotes = (len(chunks) + tamano_lote - 1) // tamano_lote

    for i in range(0, len(chunks), tamano_lote):
        lote = chunks[i : i + tamano_lote]
        num_lote = i // tamano_lote + 1
        intento = 0

        while True:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(lote, embeddings)
                else:
                    vectorstore.add_documents(lote)
                print(f"[app] Lote {num_lote}/{total_lotes} embebido "
                      f"({len(lote)} documentos).")
                break
            except Exception as exc:
                if not _es_error_rate_limit(exc) or intento >= max_reintentos:
                    raise
                espera = min(60, (2 ** intento) * 5)
                intento += 1
                print(f"[app] Rate limit alcanzado en lote {num_lote}. "
                      f"Reintentando en {espera}s (intento {intento}/{max_reintentos})...")
                time.sleep(espera)

        if num_lote < total_lotes:
            time.sleep(pausa_entre_lotes)

    return vectorstore


def build_agent(rebuild_index: bool = False):
    _validate_env()
    google_api_key = _get_google_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.1,
        google_api_key=google_api_key,
        max_retries=0,
    )
    embeddings = get_embeddings(google_api_key)

    vectorstore = _build_or_load_vectorstore(embeddings, rebuild=rebuild_index)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    web = SerpAPIWrapper()

    def nodo_agente(state: AgentState):
        prompt = f"""Eres un clasificador para el asistente virtual de NovaShop, una tienda en línea.
Decide si la pregunta debe responderse con los documentos internos (RAG) o con una búsqueda en la web.
- Usa 'RAG' para preguntas sobre políticas de la tienda: privacidad, devoluciones, reembolsos,
  envíos, entregas, términos y condiciones, o preguntas frecuentes (FAQ).
- Usa 'Web' solo para información externa o en tiempo real que no está en nuestros documentos
  (ej. tipo de cambio actual, noticias, comparación con otras tiendas).
Responde SOLO con 'RAG' o 'Web'.

Pregunta: {state['pregunta']}
"""
        decision = llm.invoke(prompt).content.strip()
        fuente = "RAG" if "RAG" in decision.upper() else "Web"
        return {"fuente": fuente}

    def nodo_rag(state: AgentState):
        docs = retriever.invoke(state["pregunta"])
        contexto = "\n\n---\n\n".join(d.page_content for d in docs)
        return {"contexto": contexto}

    def nodo_web(state: AgentState):
        try:
            resultados = web.results(state["pregunta"])
            contexto = _extraer_contexto_serpapi(resultados)
            if not contexto:
                contexto = (
                    "La búsqueda web no devolvió resultados relevantes para esta pregunta."
                )
        except Exception as exc:
            contexto = f"Ocurrió un error al realizar la búsqueda web: {exc}"
            print(f"[app] Error en nodo_web: {exc}")
        return {"contexto": contexto}

    def nodo_markdown(state: AgentState):
        prompt = f"""Eres el asistente virtual de atención al cliente de NovaShop, una tienda en línea.
Responde en español, de forma clara, directa y basada estrictamente en el contexto proporcionado.
Si el contexto no contiene la respuesta, dilo explícitamente en lugar de inventar datos.
Usa formato Markdown con viñetas cortas solo si ayudan a la claridad.

Fuente: {state['fuente']}
Contexto:
{state['contexto']}

Pregunta: {state['pregunta']}
"""
        respuesta = llm.invoke(prompt).content
        return {"respuesta": respuesta}

    def decidir_fuente(state: AgentState):
        return "if_rag" if state["fuente"] == "RAG" else "if_web"

    graph = StateGraph(AgentState)
    graph.add_node("Agente", nodo_agente)
    graph.add_node("RAG", nodo_rag)
    graph.add_node("Web", nodo_web)
    graph.add_node("Markdown", nodo_markdown)

    graph.add_edge(START, "Agente")
    graph.add_conditional_edges(
        "Agente",
        decidir_fuente,
        {
            "if_rag": "RAG",
            "if_web": "Web",
        },
    )
    graph.add_edge("RAG", "Markdown")
    graph.add_edge("Web", "Markdown")
    graph.add_edge("Markdown", END)

    return graph.compile()


def ejecutar_agente(pregunta: str, rebuild_index: bool = False) -> dict:
    agente = build_agent(rebuild_index=rebuild_index)
    return agente.invoke(
        {
            "pregunta": pregunta,
            "fuente": "",
            "contexto": "",
            "respuesta": "",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de IA sobre documentos internos (RAG + Web)")
    parser.add_argument("--pregunta", type=str, help="Pregunta para el agente")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Reconstruye el índice FAISS desde el documento fuente",
    )
    args = parser.parse_args()

    if args.pregunta:
        result = ejecutar_agente(args.pregunta, rebuild_index=args.rebuild_index)
        print("=" * 60)
        print(f"Fuente utilizada: {result['fuente']}")
        print("=" * 60)
        print(result["respuesta"])
        return

    print("Modo interactivo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tu pregunta: ").strip()
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit"}:
            break
        result = ejecutar_agente(pregunta, rebuild_index=False)
        print("=" * 60)
        print(f"Fuente utilizada: {result['fuente']}")
        print("=" * 60)
        print(result["respuesta"])
        print()


if __name__ == "__main__":
    main()