# dags/flujo_completo_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
# Importamos las tareas desde plugins/tasks/
from tasks.s3_utils import ejemplo_conexion_s3,  descargar_boletines_salta
#from tasks.procesamiento_utils import listar_pdfs_minio
from tasks.text_task import task_extract_texts
from tasks.chunk_from_txt_task import task_chunk_txt
from tasks.bm25_build_task import task_build_bm25_from_ndjson
#from tasks.bm25_query_task import task_query_bm25
#from tasks.eval_bm25_task import task_eval_bm25
#from tasks.bm25_dump_docids_task import task_dump_doc_ids
# dags/RAG_dag.py
#from tasks.make_qrels_task import task_make_qrels_from_bm25
from tasks.pinecone_upsert_task import task_pinecone_upsert
#from tasks.pinecone_query_task import task_pinecone_query
# dags/RAG_dag.py
#from tasks.fusion_rrf_task import task_fusion_query
    # dags/RAG_dag.py
#from tasks.eval_fusion_task import task_eval_fusion
    # dags/RAG_dag.py (fragmento)
from tasks.classify_chunks_agent_task import task_classify_chunks_agent
from tasks.guardrail_task import task_guardrail_chunks

# Definimos el DAG
with DAG(
    dag_id="flujo_RAG_completo",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pipeline"],
    params={                       # 👈 agrega esto
        "eval_query": "contratación pública vial"
    },
) as dag:

    descargar_boletines_task = PythonOperator(
        task_id="descargar_boletines_salta",
        python_callable=descargar_boletines_salta,
        provide_context=True,
        params={
            "year": 2025,
            "days": 14,  # ventana de 2 semanas
            "bucket_name": "respaldo2",
            "prefix": "boletines/2025/",
            "aws_conn_id": "minio_s3",
            # "url_listado": "https://... (si cambia)"
        },
    )
        
    extract_texts = PythonOperator(
        task_id="extract_texts",
        python_callable=task_extract_texts,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_pdfs": "boletines/2025/",
            "prefix_txt":  "rag/text/2025/",
            "aws_conn_id": "minio_s3",
        },
    )
    
    chunk_from_txt = PythonOperator(
        task_id="chunk_from_txt",
        python_callable=task_chunk_txt,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_txt":   "rag/text/2025/",
            "prefix_pdfs":  "boletines/2025/",
            "prefix_chunks":"rag/chunks/2025/",
            "aws_conn_id":  "minio_s3",
            "max_tokens_chunk": 300,
            "overlap": 80,
        },
    )
    
    classify_chunks_agent = PythonOperator(
        task_id="classify_chunks_agent",
        python_callable=task_classify_chunks_agent,
        op_kwargs={
            "bucket_name": "respaldo2",
            "src_prefix": "rag/chunks/2025/",
            "dst_prefix": "rag/chunks_labeled/2025/",
            "aws_conn_id": "minio_s3",
        },
    )
    
    guardrail_chunks = PythonOperator(
        task_id="guardrail_chunks",
        python_callable=task_guardrail_chunks,
        op_kwargs={
            "bucket_name": "respaldo2",
            "in_prefix":  "rag/chunks_labeled/2025/",
            "out_prefix": "rag/chunks_curated/2025/",
            "aws_conn_id": "minio_s3",
        },
    )

    build_bm25 = PythonOperator(
        task_id="build_bm25",
        python_callable=task_build_bm25_from_ndjson,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_chunks": "rag/chunks_curated/2025/",
            "prefix_models": "rag/models/2025/",
            "aws_conn_id": "minio_s3",
        },
    )

    pinecone_upsert = PythonOperator(
        task_id="pinecone_upsert",
        python_callable=task_pinecone_upsert,
        op_kwargs={
            "bucket_name":  "respaldo2",
            "prefix_chunks":"rag/chunks_curated/2025/",
            "aws_conn_id":  "minio_s3",
            "index_name":   "boletines-2025",
            "namespace":    "2025",
            "model_name":   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "batch_size":   128,
        },
    )   

    


descargar_boletines_task >> extract_texts >> chunk_from_txt >> classify_chunks_agent >> guardrail_chunks

guardrail_chunks >> build_bm25
guardrail_chunks >> pinecone_upsert

