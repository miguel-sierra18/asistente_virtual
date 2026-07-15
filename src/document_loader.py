"""
document_loader.py
-------------------
Módulo encargado de cargar y procesar documentos (PDF o CSV) para
convertirlos en objetos `Document` de LangChain, listos para ser
divididos en fragmentos (chunks) y luego indexados en un vector store.

Soporta:
- CSV: cada fila se convierte en un Document independiente (ideal para
  datos tabulares como ventas de productos).
- PDF: el texto se extrae por página y luego se divide en fragmentos
  más pequeños con un RecursiveCharacterTextSplitter, para que el
  contexto que se le pasa al LLM sea manejable.

Uso:
    from document_loader import cargar_documento

    documentos = cargar_documento("data/ventas_productos.csv")
    for doc in documentos[:3]:
        print(doc.page_content)
        print(doc.metadata)
"""

from pathlib import Path
from typing import List

import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Si un CSV tiene más filas que este umbral, se agrega automáticamente
# (por columna categórica + mes) para no exceder el límite de peticiones
# por minuto del plan gratuito de embeddings de Gemini (~100 req/min).
UMBRAL_AGREGACION_CSV = 500


# Configuración por defecto del divisor de texto para PDFs.
# chunk_size: cuántos caracteres aproximados tiene cada fragmento.
# chunk_overlap: cuántos caracteres se repiten entre fragmentos
# consecutivos, para no perder contexto en los cortes.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _cargar_csv(ruta: Path) -> List[Document]:
    """
    Carga un archivo CSV y convierte cada fila en un Document.
    CSVLoader ya arma automáticamente el page_content con el formato
    "columna: valor" para cada fila, lo cual es muy legible para el LLM.
    """
    loader = CSVLoader(
        file_path=str(ruta),
        encoding="utf-8",
        csv_args={"delimiter": ","},
    )
    documentos = loader.load()

    # Enriquecemos los metadatos para poder filtrar/depurar después
    # (por ejemplo, saber de qué archivo y fila vino cada Document).
    for i, doc in enumerate(documentos):
        doc.metadata["source_file"] = ruta.name
        doc.metadata["tipo"] = "csv"

    return documentos


def _detectar_columnas_csv(df: pd.DataFrame) -> dict:
    """
    Detecta heurísticamente qué columnas usar para agregar el CSV:
    una columna de fecha, una categórica principal (ej. Producto) y,
    opcionalmente, columnas numéricas para sumar (ej. Cantidad, Total_Venta).
    """
    columna_fecha = None
    for col in df.columns:
        if "fecha" in col.lower() or "date" in col.lower():
            columna_fecha = col
            break

    columnas_numericas = df.select_dtypes(include="number").columns.tolist()

    columna_categorica = None
    for col in df.columns:
        if col == columna_fecha or col in columnas_numericas:
            continue
        # Elegimos la primera columna de texto con cardinalidad razonable
        # (ni un identificador único ni una constante).
        n_unicos = df[col].nunique()
        if 1 < n_unicos < len(df):
            columna_categorica = col
            break

    return {
        "fecha": columna_fecha,
        "categorica": columna_categorica,
        "numericas": columnas_numericas,
    }


def _cargar_csv_agregado(ruta: Path) -> List[Document]:
    """
    Agrega un CSV grande por (columna categórica principal + mes) para
    reducir drásticamente el número de documentos a vectorizar, evitando
    el error 429 (RESOURCE_EXHAUSTED) del plan gratuito de embeddings.

    Cada Document resultante resume un período: totales, promedios y
    número de registros, en lugar de una fila cruda por documento.
    """
    df = pd.read_csv(ruta)
    cols = _detectar_columnas_csv(df)

    if not cols["fecha"] or not cols["categorica"]:
        # No se pudo detectar una estructura agregable; caemos al modo fila a fila.
        print("[document_loader] No se detectaron columnas de fecha/categoría "
              "claras; se usará carga fila por fila (puede exceder el rate limit).")
        return _cargar_csv(ruta)

    df[cols["fecha"]] = pd.to_datetime(df[cols["fecha"]], errors="coerce")
    df["_periodo"] = df[cols["fecha"]].dt.to_period("M").astype(str)

    # Columnas tipo "precio unitario" o "promedio" se promedian; el resto
    # (cantidades, totales) se suman. Sumar un precio unitario no tiene
    # sentido de negocio, así que lo detectamos por nombre.
    palabras_promedio = ("precio", "unitario", "promedio", "avg", "price")
    agg_dict = {}
    for col in cols["numericas"]:
        if any(p in col.lower() for p in palabras_promedio):
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "sum"

    resumen = (
        df.groupby([cols["categorica"], "_periodo"])
        .agg(agg_dict)
        .reset_index()
    )
    conteo = df.groupby([cols["categorica"], "_periodo"]).size().reset_index(name="_num_registros")
    resumen = resumen.merge(conteo, on=[cols["categorica"], "_periodo"])

    documentos: List[Document] = []
    for _, fila in resumen.iterrows():
        partes = [
            f"{cols['categorica']}: {fila[cols['categorica']]}",
            f"Periodo: {fila['_periodo']}",
        ]
        for col in cols["numericas"]:
            etiqueta = "promedio" if agg_dict[col] == "mean" else "suma"
            valor = round(fila[col], 2) if isinstance(fila[col], float) else fila[col]
            partes.append(f"{col} ({etiqueta} del periodo): {valor}")
        partes.append(f"Número de registros agregados: {fila['_num_registros']}")

        contenido = "\n".join(partes)
        documentos.append(
            Document(
                page_content=contenido,
                metadata={
                    "source_file": ruta.name,
                    "tipo": "csv_agregado",
                    cols["categorica"]: str(fila[cols["categorica"]]),
                    "periodo": fila["_periodo"],
                },
            )
        )

    print(f"[document_loader] CSV agregado por '{cols['categorica']}' + mes: "
          f"{len(df)} filas originales -> {len(documentos)} documentos.")

    return documentos


def _cargar_pdf(ruta: Path) -> List[Document]:
    """
    Carga un PDF, extrae el texto página por página y luego lo divide
    en fragmentos más pequeños para facilitar la recuperación semántica.
    """
    loader = PyPDFLoader(str(ruta))
    paginas = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documentos = splitter.split_documents(paginas)

    for doc in documentos:
        doc.metadata["source_file"] = ruta.name
        doc.metadata["tipo"] = "pdf"

    return documentos


def cargar_documento(ruta_archivo: str, modo_csv: str = "auto") -> List[Document]:
    """
    Punto de entrada principal. Detecta la extensión del archivo
    (.csv o .pdf) y aplica la estrategia de carga correspondiente.

    modo_csv (solo aplica a archivos .csv):
        - "auto": agrega automáticamente si el CSV supera
          UMBRAL_AGREGACION_CSV filas; si no, carga fila por fila.
        - "agregado": fuerza la agregación por categoría + mes,
          sin importar el tamaño del archivo.
        - "detallado": fuerza la carga fila por fila (una fila = un Document).

    Lanza ValueError si la extensión o el modo no están soportados, y
    FileNotFoundError si el archivo no existe.
    """
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    extension = ruta.suffix.lower()

    if extension == ".csv":
        if modo_csv not in {"auto", "agregado", "detallado"}:
            raise ValueError("modo_csv debe ser 'auto', 'agregado' o 'detallado'")

        if modo_csv == "agregado":
            documentos = _cargar_csv_agregado(ruta)
        elif modo_csv == "detallado":
            documentos = _cargar_csv(ruta)
        else:  # auto
            n_filas = sum(1 for _ in open(ruta, encoding="utf-8")) - 1  # -1 por el header
            if n_filas > UMBRAL_AGREGACION_CSV:
                print(f"[document_loader] CSV con {n_filas} filas supera el umbral "
                      f"({UMBRAL_AGREGACION_CSV}); se agregará automáticamente por "
                      f"categoría + mes para respetar el rate limit gratuito.")
                documentos = _cargar_csv_agregado(ruta)
            else:
                documentos = _cargar_csv(ruta)
    elif extension == ".pdf":
        documentos = _cargar_pdf(ruta)
    else:
        raise ValueError(
            f"Extensión '{extension}' no soportada. Usa .csv o .pdf"
        )

    print(f"[document_loader] '{ruta.name}' cargado correctamente: "
          f"{len(documentos)} fragmentos generados.")

    return documentos


def cargar_coleccion(directorio: str, modo_csv: str = "auto") -> List[Document]:
    """
    Carga TODOS los documentos soportados (.csv y .pdf) dentro de una
    carpeta, aplicando cargar_documento() a cada uno y fusionando los
    resultados en una sola lista de Documents lista para indexar.

    Cada Document conserva en sus metadatos ('source_file') a qué
    archivo original pertenece, así que el agente puede citar la fuente
    exacta aunque la colección tenga varios documentos mezclados.

    Lanza FileNotFoundError si la carpeta no existe, y no falla si
    algún archivo individual da error: lo reporta y continúa con los demás.
    """
    carpeta = Path(directorio)
    if not carpeta.exists() or not carpeta.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta: {directorio}")

    archivos = sorted(
        [f for f in carpeta.iterdir() if f.suffix.lower() in (".csv", ".pdf")]
    )

    if not archivos:
        raise RuntimeError(
            f"No se encontraron archivos .csv o .pdf en '{directorio}'"
        )

    todos_los_documentos: List[Document] = []
    for archivo in archivos:
        try:
            docs = cargar_documento(str(archivo), modo_csv=modo_csv)
            todos_los_documentos.extend(docs)
        except Exception as e:
            print(f"[document_loader] ⚠️ Error cargando '{archivo.name}', "
                  f"se omite este archivo: {e}")

    print(f"[document_loader] Colección cargada: {len(archivos)} archivo(s) -> "
          f"{len(todos_los_documentos)} documentos totales.")

    return todos_los_documentos


if __name__ == "__main__":
    # Prueba rápida: carga el documento de sugerencia y muestra
    # una vista previa de los primeros fragmentos.
    ruta_ejemplo = "data/ventas_productos.csv"
    docs = cargar_documento(ruta_ejemplo)

    print(f"\nTotal de fragmentos: {len(docs)}")
    print("\n--- Vista previa de los primeros 2 fragmentos ---\n")
    for doc in docs[:2]:
        print("Contenido:")
        print(doc.page_content)
        print("Metadatos:", doc.metadata)
        print("-" * 50)