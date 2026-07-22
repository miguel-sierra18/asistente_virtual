# 🛒 Asistente Virtual — NovaShop

Agente de IA con arquitectura RAG (Retrieval-Augmented Generation) capaz de responder
preguntas sobre la documentación interna de una tienda en línea ficticia (NovaShop),
y de recurrir a una búsqueda web cuando la pregunta no puede resolverse con esos
documentos. Construido con **LangChain**, **LangGraph**, **FAISS** y la **API de
Google Gemini**, con una interfaz de chat en **Streamlit**.

Este proyecto forma parte de un desafío de aprendizaje enfocado en tres etapas:
1. Cargar y procesar documentos (PDF/CSV) para que la aplicación entienda su contenido.
2. Construir un agente de IA capaz de responder preguntas sobre esos documentos.
3. Desplegar el agente en la nube para que sea accesible públicamente.

---

## 📋 Descripción general

NovaShop es una tienda en línea ficticia. El asistente responde preguntas de
clientes sobre:

- **Política de privacidad** — qué datos se recopilan, con quién se comparten, derechos ARCO.
- **Política de reembolsos y devoluciones** — plazos, condiciones, garantías.
- **Preguntas frecuentes (FAQ)** — métodos de pago, cuentas, cancelaciones, soporte.
- **Guía de envíos y entregas** — costos, tiempos, cobertura, paqueterías.
- **Términos y condiciones** — precios, propiedad intelectual, jurisdicción.

Cuando una pregunta requiere información que **no** está en estos documentos
(por ejemplo, el tipo de cambio del día), el agente decide automáticamente
consultar la web en lugar de inventar una respuesta.

---

## 🏗️ Arquitectura de la solución

El agente está implementado como un grafo de estados con **LangGraph**:

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Agente    │  ← el LLM clasifica la pregunta
                    │(clasificador)│
                    └──────┬──────┘
                 ┌─────────┴─────────┐
                 ▼                   ▼
           ┌───────────┐       ┌───────────┐
           │    RAG    │       │    Web    │
           │ (FAISS +  │       │ (SerpAPI) │
           │ embeddings)│      │           │
           └─────┬─────┘       └─────┬─────┘
                 └─────────┬─────────┘
                           ▼
                    ┌─────────────┐
                    │  Markdown   │  ← el LLM redacta la respuesta final
                    │ (respuesta) │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │     END     │
                    └─────────────┘
```

**Flujo paso a paso:**

1. **Carga y procesamiento de documentos** (`src/document_loader.py`): detecta si
   cada archivo de la colección es PDF o CSV.
   - PDF → se extrae el texto por página y se divide en fragmentos con
     `RecursiveCharacterTextSplitter` (chunks de ~1000 caracteres).
   - CSV → cada fila se convierte en un `Document`; si el archivo supera 500
     filas, se agrega automáticamente por categoría + mes para no exceder el
     límite de peticiones por minuto del plan gratuito de embeddings.
2. **Indexación**: los fragmentos se convierten en vectores con
   `GoogleGenerativeAIEmbeddings` y se guardan en un índice **FAISS** local
   (se construye en lotes con reintentos automáticos ante error 429).
3. **Nodo Agente (clasificador)**: el LLM decide si la pregunta debe
   responderse con los documentos internos (`RAG`) o con una búsqueda web (`Web`).
4. **Nodo RAG**: recupera los fragmentos más relevantes del índice FAISS
   (`k=6`) y arma el contexto.
5. **Nodo Web**: consulta SerpAPI y extrae el `answer_box`, `knowledge_graph`
   y los primeros `organic_results` como contexto.
6. **Nodo Markdown**: el LLM redacta la respuesta final en español, basada
   estrictamente en el contexto recuperado — si el contexto no contiene la
   respuesta, lo dice explícitamente en lugar de inventar datos.

---

## 🛠️ Tecnologías y herramientas

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| Orquestación del agente | LangChain + LangGraph |
| LLM y embeddings | Google Gemini API (`langchain-google-genai`) |
| Vector store | FAISS (`faiss-cpu`) |
| Búsqueda web | SerpAPI |
| Carga de documentos | `pypdf`, `pandas` |
| Interfaz | Streamlit |
| Configuración | `python-dotenv` |
| Contenerización | Docker / Docker Compose |
| Deploy | Streamlit Community Cloud (también documentado: OCI Compute + Docker) |

---

## 🚀 Instrucciones para ejecutar el proyecto

### 1. Requisitos previos

- Python 3.11 o superior
- Una API key de [Google AI Studio](https://aistudio.google.com/apikey) (Gemini, tier gratuito)
- Una API key de [SerpAPI](https://serpapi.com) (tier gratuito, 100 búsquedas/mes)

### 2. Instalación

```bash
git clone <tu-repositorio>
cd agente-documental
python3 -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y agrega tus claves reales:

```
GEMINI_API_KEY=tu_key_real
SERPAPI_API_KEY=tu_key_real
```

### 4. Ejecutar por línea de comandos

```bash
python app.py --pregunta "¿Cuántos días tengo para devolver un producto?" --rebuild-index
```

(`--rebuild-index` solo es necesario la primera vez, o cuando cambias los documentos en `data/documentos/`.)

### 5. Ejecutar la interfaz web

```bash
streamlit run streamlit_app.py
```

Se abre en `http://localhost:8501`. Desde el sidebar puedes:
- Subir nuevos documentos (PDF/CSV) a la colección.
- Eliminar documentos existentes.
- Reconstruir el índice FAISS manualmente.
- Activar el "Modo debug" para ver el contexto crudo usado en cada respuesta.

### 6. Deploy

Este repositorio incluye dos guías completas:
- [`DEPLOY_STREAMLIT_CLOUD.md`](./DEPLOY_STREAMLIT_CLOUD.md) — deploy gratuito en Streamlit Community Cloud (recomendado, sin Docker).
- [`DEPLOY_OCI.md`](./DEPLOY_OCI.md) — deploy en una VM Always Free de Oracle Cloud Infrastructure con Docker.

---

## 💬 Ejemplos de preguntas que el agente puede responder

**Sobre políticas internas (vía RAG):**
- ¿Cuántos días tengo para devolver un producto?
- ¿Qué métodos de pago aceptan?
- ¿Cuánto cuesta el envío estándar y en cuánto tiempo llega?
- ¿NovaShop vende mis datos a terceros?
- ¿Qué pasa si mi producto llega dañado?
- ¿Puedo devolver un producto en liquidación final?
- ¿NovaShop tiene tiendas físicas?

**Fuera del alcance de los documentos (vía búsqueda Web):**
- ¿Cuál es el tipo de cambio del dólar hoy?
- ¿Qué tendencias hay en el mercado de e-commerce en 2026?

---

## 🗨️ Ejemplos de respuestas generadas por el agente

> **Pregunta:** ¿Cuántos días tengo para devolver un producto?
>
> **Fuente usada:** RAG
>
> Tienes **30 días naturales** a partir de la fecha de entrega para solicitar
> la devolución de un producto. Pasado este plazo, no se pueden procesar
> devoluciones ni reembolsos, salvo defectos de fábrica cubiertos por
> garantía (12 meses en electrónica, 24 meses en mobiliario).

> **Pregunta:** ¿Qué métodos de pago aceptan?
>
> **Fuente usada:** RAG
>
> NovaShop acepta:
> - Tarjetas de crédito y débito Visa, Mastercard y American Express.
> - PayPal.
> - Pago en efectivo en tiendas OXXO (el pedido se reserva 48 horas).
> - Meses sin intereses (3, 6 o 12) en compras mayores a $999 MXN con tarjetas participantes.

> **Pregunta:** ¿NovaShop hace envíos a Guatemala?
>
> **Fuente usada:** RAG
>
> No. Actualmente NovaShop solo realiza envíos dentro de territorio
> mexicano; no ofrece envíos internacionales por el momento.

> **Pregunta:** ¿Cuál es el tipo de cambio del dólar hoy?
>
> **Fuente usada:** Web
>
> *(Respuesta generada en tiempo real a partir de los resultados de búsqueda
> de SerpAPI — el valor exacto varía según el día de la consulta.)*

---

## Pruebas
![Texto alternativo](<img width="1918" height="967" alt="deploy_streamlit" src="https://github.com/user-attachments/assets/1d8a6b8b-f921-4926-8736-9d62e0ce4f45" />
)

---

## Enlace del proyecto

[Deploy en streamlit](https://asistentevirtual-fmxyzqnjaevqzdjqxqftq4.streamlit.app/)

---

## 📁 Estructura del proyecto

```
agente-documental/
├── app.py                   # Lógica del agente (LangGraph + FAISS + Gemini + SerpAPI)
├── streamlit_app.py         # Interfaz de chat en Streamlit
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── DEPLOY_OCI.md
├── DEPLOY_STREAMLIT_CLOUD.md
├── data/
│   └── documentos/          # Colección de PDF/CSV sobre los que responde el agente
│       ├── politica_privacidad.pdf
│       ├── politica_reembolsos_devoluciones.pdf
│       ├── preguntas_frecuentes.pdf
│       ├── guia_envios_entregas.pdf
│       └── terminos_condiciones.pdf
└── src/
    └── document_loader.py   # Carga y procesamiento de PDF/CSV (individual o colección)
```
