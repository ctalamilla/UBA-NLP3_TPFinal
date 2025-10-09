# RAG – Boletín Oficial (Operativo)

Sistema RAG híbrido (BM25 + Pinecone + LLM) para consultar el Boletín Oficial de Salta.  
Incluye microservicios: **MinIO (S3)**, **Airflow (ingesta/indexación)**, **FastAPI (API RAG)** y **Streamlit (UI)**.

# Estructura de archivos
```text
.
├─ dags/
│  └─ Ingestion_pipeline.py        # DAG de Airflow que orquesta el flujo ETL/RAG completo.
│                                  # Llama a tasks/* (extract > chunk > index > eval).
├─ datalake/                        # Carpeta local (opcional) para staging; en prod se usa S3/MinIO.
├─ docker-compose.yaml              # Levanta Airflow, FastAPI, Streamlit, MinIO, etc.
├─ Dockerfile                       # Imagen base (raíz) – útil para notebooks o jobs sueltos.

├─ fastapi_app/
│  ├─ Dockerfile                    # Imagen del microservicio de consulta.
│  ├─ main.py                       # Endpoints (/health, /ask, /query, /vector/query, /eval/*).
│  ├─ performance.py                # Endpoints/útiles para medir latencias y throughput.
│  ├─ pipeline.py                   # Pipeline RAG híbrido (BM25 + Pinecone + RRF + LLM).
│  ├─ requirements.txt
│  ├─ s3_boto.py                    # Cliente S3/MinIO para leer bm25.pkl y NDJSON de chunks.
│  └─ vector_pinecone_api.py        # Helper de Pinecone (ensure_index, query con embeddings).

├─ frontend_streamlit/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ streamlit_app.py              # UI simple para probar RAG (muestra chunk_id/source/score/…).

├─ mlflow/
│  └─ artifacts/                    # Artefactos de experimentos (si usás MLflow localmente).
├─ mlruns/                           # Metadatos de MLflow (experimentos, runs).

├─ notebooks/
│  └─ pipeline_rag.ipynb            # Notebook de prototipado: chunking, fusión RRF, pruebas locales.

├─ plugins/
│  ├─ __init__.py
│  └─ tasks/                        # “Librería de tareas” usada por Airflow y scripts.
│     ├─ agent_classifier.py        # Clasificador LLM/heurístico por OP (categoría, extraídos).
│     ├─ agente_verificador.py      # Verificaciones/guardrails con LLM (si aplica).
│     ├─ bm25_build_task.py         # Construye índice BM25 (bm25.pkl) desde NDJSON.
│     ├─ bm25_dump_docids_task.py   # Exporta mapping de doc_ids (debug/diagnóstico).
│     ├─ bm25_index.py              # Implementación BM25Index (search, doc_ids, persistencia).
│     ├─ bm25_query_task.py         # Consulta de prueba contra el BM25.
│     ├─ chunk_from_txt_task_op.py  # Chunker “OP-first”: agrupa por boletín/op y emite NDJSON.
│     ├─ chunk_from_txt_task.py     # Chunker “plain”: un NDJSON por PDF base (modo legacy).
│     ├─ classify_chunks_agent_task.py # Clasifica chunks ya generados (etiquetas).
│     ├─ classify_op_texts_task.py  # Extrae + clasifica cada OP desde TXT crudo.
│     ├─ documents.py               # Clase Document + chunk_text (lógica de segmentación).
│     ├─ eval_bm25_task.py          # Métricas sobre BM25 (recall@k, mrr, etc.).
│     ├─ eval_fusion_task.py        # Eval de fusión RRF (AP@k, nDCG@k, Recall@k, MRR).
│     ├─ extract_texts_by_op_task.py# Crea text_op/ y text_op_meta/ desde PDFs o fuentes.
│     ├─ fusion_rrf_task.py         # Ejecuta RRF (BM25 + vector) y guarda resultados en rag/fusion/.
│     ├─ fusion.py                  # Implementación pura de RRF (función rrf_combine).
│     ├─ guardrail_task.py          # Detección de prompt-injection/ruido en chunks.
│     ├─ io_utils.py                # Utilidades de IO locales.
│     ├─ loader_pdfs.py             # PDF→Document (limpieza, normalización, dehyphen, etc.).
│     ├─ make_qrels_task.py         # Construye qrels.csv (ground truth) para evaluación.
│     ├─ metrics.py                 # Métricas de ranking (AP, nDCG, Recall, MRR…).
│     ├─ pinecone_query_task.py     # Consulta vectorial de prueba (top_k) desde Airflow/script.
│     ├─ pinecone_upsert_op_task.py # Upsert a Pinecone desde NDJSON “OP-first” (con metadatos extra).
│     ├─ pinecone_upsert_task.py    # Upsert a Pinecone desde NDJSON legacy.
│     ├─ procesamiento_utils.py     # Limpieza, normalización, helpers de texto.
│     ├─ qrels_utils.py             # Lectura/parseo de qrels.
│     ├─ s3_utilities.py            # Listar/leer/subir a S3 (read_text, upload_json, list_keys…).
│     ├─ s3_utils.py                # Compat/atajos S3 (antiguo).
│     ├─ text_task.py               # Tareas misceláneas de texto.
│     ├─ utils_op_split.py          # Split de documentos en OPs (nombres, patrones).
│     ├─ vector_pinecone_op.py      # Cliente Pinecone orientado a OP (IDs, namespaces).
│     └─ vector_pinecone.py         # Cliente Pinecone genérico (ensure_index, upsert, query).

├─ rag_notebook/
│  ├─ Dockerfile
│  └─ requirements.txt              # Entorno liviano para reproducir el notebook.

└─ README.md                        # Cómo correr, variables de entorno, flujo E2E.

```

## 1) Requisitos

- Docker + Docker Compose
- (Opcional) `jq` para formatear respuestas en consola

## 2) Variables de entorno (`.env`)

Crea un archivo `.env` en el root del repo con algo así:

```bash
# --- MinIO / S3 ---
S3_ENDPOINT_URL=http://minio:9000          # dentro de compose
S3_ENDPOINT_URL_HOST=http://localhost:9000  # si accedés desde tu host
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1
S3_BUCKET=respaldo2
S3_USE_SSL=false

# --- OpenAI ---
OPENAI_API_KEY=sk-xxx

# --- Pinecone ---
PINECONE_API_KEY=pc-xxx
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX=boletines-2025
PINECONE_NAMESPACE=2025
EMB_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# --- RAG modelos / paths en S3 ---
BM25_MODEL_KEY=rag/models/2025/bm25.pkl
CHUNKS_PREFIX=rag/chunks_op/2025/

# --- Puertos (si tu compose los usa) ---
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
AIRFLOW_WEB_PORT=8080
```

> Asegurate de que **exista el bucket `respaldo2`** en MinIO y que estén subidos:
> - `rag/models/2025/bm25.pkl`
> - `rag/chunks_op/2025/*.ndjson`  
> (los genera tu pipeline de ingesta / DAGs)

## 3) Levantar los servicios

### 3.1 Docker Compose (todo junto)

```bash
docker compose up -d --build
```

Servicios típicos:
- **MinIO API**: http://localhost:9000  
- **MinIO Console**: http://localhost:9001 (user/pass = `minioadmin` / `minioadmin` salvo que lo cambies)
- **Airflow Web**: http://localhost:8080
- **FastAPI**: http://localhost:8000
- **Streamlit**: http://localhost:8501

> Si tu `docker-compose.yaml` trae los servicios por separado, podés correrlos selectivamente, ej.  
> `docker compose up -d minio airflow fastapi streamlit`.

### 3.2 Inicializar Airflow (una sola vez)

Si tu compose sigue el patrón oficial, algo como:

```bash
# inicializa DB y crea usuario admin
docker compose run --rm airflow-webserver airflow db init
docker compose run --rm airflow-webserver   airflow users create --username admin --password admin   --firstname Admin --lastname User --role Admin --email admin@example.com
```

Entrá a **http://localhost:8080**, logueate y **habilitá el DAG** `dags/Ingestion_pipeline.py`.

#### Conexión a MinIO en Airflow
En **Admin → Connections** crea (o edita) una conexión:
- Conn Id: `minio_s3`  (o el que uses en tus tasks)
- Conn Type: **Amazon Web Services**
- Extra (JSON):
  ```json
  {
    "endpoint_url": "http://minio:9000",
    "region_name": "us-east-1",
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin"
  }
  ```

> El DAG escribe/lee en `s3://respaldo2/…` y deja todo listo para la API.

## 4) FastAPI – Endpoints y prueba rápida

Health:

```bash
curl -s http://localhost:8000/health | jq
```

Consulta RAG (usa BM25 + Pinecone + LLM):

```bash
curl -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{
    "query": "CASTRO, MARIA EUGENIA en que OP y boletin se menciona?",
    "k_final": 2
  }' | jq '.performance'
```

Ejemplo real de performance devuelto por la API:

```json
{
  "total_time": 11.945,
  "breakdown": {
    "bm25_time": 0.014,
    "vector_time": 1.132,
    "fusion_time": 0.0,
    "candidates_time": 0.095,
    "rerank_time": 0.0,
    "llm_summary_time": 3.99,
    "llm_answer_time": 2.745,
    "llm_verification_time": 3.967
  },
  "tokens": { "...": "..." },
  "cost": { "total_usd": 0.000704 }
}
```

Para ver metadatos del primer resultado:

```bash
curl -s -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{
    "query": "CASTRO, MARIA EUGENIA en que OP y boletin se menciona?",
    "k_final": 2
  }' | jq '.results[0].metadata'
```

Ejemplo de respuesta (real):

```json
{
  "chunk_id": "22044_2025-10-02::p1::2",
  "source": "boletines/2025/22044_2025-10-02.pdf",
  "boletin": "22044",
  "fecha": "2025-10-02",
  "op": "100128767",
  "ndjson_key": "rag/chunks_op/2025/22044_2025-10-02.ndjson",
  "meta": { "...": "..." }
}
```

Otro test de la API (con verificación LLM incluida):

```bash
curl -s -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{
    "query": "GLATSTEIN, JULIO en que OP se menciona y en que boletin?",
    "k_final": 2
  }' | jq '{answer, verification}'
```

Ejemplo de salida:

```json
{
  "answer": "GLATSTEIN, JULIO se menciona en el Boletín N° 22042, publicado el 30 de septiembre de 2025, en la OP N° 100128809. Según el contexto, se ordena la publicación de edictos ...",
  "verification": "✅ TOTALMENTE RESPALDADA ..."
}
```

> Endpoints extra:  
> `GET /ask?q=...` (modo rápido)  
> `POST /vector/query` (consulta directa al índice vectorial)

## 5) Streamlit – UI

Si está en `docker-compose.yaml`, con el `up -d` ya queda online en:  
**http://localhost:8501**

Si querés correrlo manual:

```bash
# build
docker build -t boletin-streamlit ./frontend_streamlit
# run
docker run --rm -p 8501:8501 --env-file .env   -e FASTAPI_BASE_URL=http://host.docker.internal:8000   boletin-streamlit
```

> `FASTAPI_BASE_URL` debe apuntar al host donde corre la API (dentro del compose podés usar el nombre del servicio, p.ej. `http://fastapi:8000`).

## 6) Estructura de datos en MinIO (resumen)

```text
rag/
├─ chunks_op/              # NDJSON de chunks por boletín/OP (pipeline automático)
│   └─ 2025/
│      ├─ 22043_2025-10-01.ndjson
│      └─ …
├─ chunks_op_curated/      # NDJSON “curados” (correcciones/merge manual)
│   └─ 2025/
├─ fusion/                 # Resultados de búsqueda híbrida (RRF) por consulta
│   └─ 2025/
│      └─ fusion_<consulta_sanitizada>_<timestamp>.json
├─ metrics/                # Reportes de evaluación (AP@k, nDCG@k, Recall@k, MRR)
│   └─ 2025/
│      └─ fusion_eval_<consulta>_<timestamp>.json
├─ models/                 # Artefactos de indexación
│   └─ 2025/
│      └─ bm25.pkl
├─ qrels/                  # Relevancias “verdad terreno” para evaluación
│   └─ 2025/
│      └─ qrels.csv
├─ text_op/                # TXT crudos por OP (post extracción)
│   └─ 2025/
│      └─ 22047_OP100129085_2025-10-07.txt
├─ text_op_meta/           # Metadatos por OP (JSON) alineados a los TXT
│   └─ 2025/
│      └─ 22047_OP100129085_2025-10-07.json
└─ notebook_sanity.txt     # Marcador/sanity del entorno

```

## 7) Operación diaria

1. **Ingesta** (Airflow): corre `Ingestion_pipeline.py`, genera `*.ndjson` y actualiza `bm25.pkl`/Pinecone.  
2. **API**: expone `/query` con recuperación híbrida + respuesta LLM + verificación.  
3. **UI**: Streamlit llama a FastAPI para consultas en NL.  
4. **Monitoreo**: revisar `performance` en la respuesta y logs de cada servicio.

## 8) Problemas frecuentes

- **`⚠️ OPENAI_API_KEY no configurada`** → setear `OPENAI_API_KEY` en `.env`.  
- **Pinecone “dimension mismatch”** → el índice debe ser **384** (modelo MiniLM).  
- **No hay resultados** → validar que existan `*.ndjson` en `CHUNKS_PREFIX` y que `bm25.pkl` esté en `BM25_MODEL_KEY`.  
- **Conexión MinIO desde la API** → si corrés local, usá `S3_ENDPOINT_URL=http://minio:9000` dentro de compose; desde el host, `http://localhost:9000`.

### Comandos útiles

Ver estado de los servicios:
```bash
docker compose ps
docker compose logs -f fastapi
docker compose logs -f airflow-webserver
docker compose logs -f streamlit
```

Recrear sólo la API:
```bash
docker compose up -d --build fastapi
```
