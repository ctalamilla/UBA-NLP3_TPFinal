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
# dags/flujo_RAG.py (o el que uses)
from tasks.extract_texts_by_op_task import task_extract_texts_by_op
from tasks.classify_op_texts_task import task_classify_op_texts
from tasks.chunk_from_txt_task_op import task_chunk_txt_op
from tasks.pinecone_upsert_op_task import task_pinecone_upsert_op
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
        
   
    extract_texts_by_op = PythonOperator(
        task_id="extract_texts_by_op",
        python_callable=task_extract_texts_by_op,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_pdfs": "boletines/2025/",
            "prefix_txt":  "rag/text_op/2025/",
            # opcional; si no lo pasás se auto-deriva a rag/text_op_meta/2025/
            "prefix_meta": "rag/text_op_meta/2025/",
            "aws_conn_id": "minio_s3",
            "ignore_first_pages": 2,
            "ignore_last_pages": 1,
        },
    )
    
    classify_op_txts = PythonOperator(
        task_id="classify_op_texts",
        python_callable=task_classify_op_texts,
        op_kwargs={
            "bucket_name": "respaldo2",
            # no pasamos meta_key_in -> usa modo sidecars
            "meta_prefix": "rag/text_op_meta/2025/",
            "aws_conn_id": "minio_s3",
            "use_llm": True,                  # o False si querés solo heurística
            "model_name": "gpt-4o-mini",
            "force_overwrite": False,         # True para re-escribir si ya hay clasificación
        },
    )

    chunk_from_txt_op = PythonOperator(
        task_id="chunk_from_txt_op",
        python_callable=task_chunk_txt_op,
        op_kwargs={
            "bucket_name":  "respaldo2",
            "prefix_txt":   "rag/text_op/2025/",        # TXT por OP
            "prefix_pdfs":  "boletines/2025/",          # para 'source' correcto
            "prefix_chunks":"rag/chunks_op/2025/",      # salida NDJSON por PDF
            "meta_prefix":  "rag/text_op_meta/2025/",   # sidecars y/o enriquecidos
            "aws_conn_id":  "minio_s3",
            "max_tokens_chunk": 400,
            "overlap": 120,
        },
    )

    
    build_bm25 = PythonOperator(
        task_id="build_bm25",
        python_callable=task_build_bm25_from_ndjson,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_chunks": "rag/chunks_op/2025/",#"rag/chunks_curated/2025/",
            "prefix_models": "rag/models/2025/",
            "aws_conn_id": "minio_s3",
        },
    )


    pinecone_upsert_op = PythonOperator(
        task_id="pinecone_upsert_op",
        python_callable=task_pinecone_upsert_op,
        op_kwargs={
            "bucket_name":   "respaldo2",
            "prefix_chunks": "rag/chunks_op/2025/",   # <- el prefijo del pipeline por-OP
            "aws_conn_id":   "minio_s3",
            "index_name":    "boletines-2025",
            "namespace":     "2025",
            "model_name":    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "batch_size":    128,
            "min_chars":     20,
        },
    )
    


#descargar_boletines_task >> extract_texts >> chunk_from_txt >> classify_chunks_agent >> guardrail_chunks
descargar_boletines_task >> extract_texts_by_op >> classify_op_txts >> chunk_from_txt_op

chunk_from_txt_op >> build_bm25
chunk_from_txt_op >> pinecone_upsert_op

#guardrail_chunks >> build_bm25
#guardrail_chunks >> pinecone_upsert

