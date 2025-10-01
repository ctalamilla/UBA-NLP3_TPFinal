# dags/flujo_completo_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
# Importamos las tareas desde plugins/tasks/
from tasks.s3_utils import ejemplo_conexion_s3,  descargar_boletines_salta
from tasks.procesamiento_utils import listar_pdfs_minio
from tasks.text_task import task_extract_texts
from tasks.chunk_from_txt_task import task_chunk_txt
from tasks.bm25_build_task import task_build_bm25_from_ndjson
from tasks.bm25_query_task import task_query_bm25
from tasks.eval_bm25_task import task_eval_bm25
from tasks.bm25_dump_docids_task import task_dump_doc_ids
# dags/RAG_dag.py
from tasks.make_qrels_task import task_make_qrels_from_bm25
from tasks.pinecone_upsert_task import task_pinecone_upsert
from tasks.pinecone_query_task import task_pinecone_query
# dags/RAG_dag.py
from tasks.fusion_rrf_task import task_fusion_query
    # dags/RAG_dag.py
from tasks.eval_fusion_task import task_eval_fusion



# Definimos el DAG
with DAG(
    dag_id="flujo_RAG",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pipeline"],
    params={                       # 👈 agrega esto
        "eval_query": "contratación pública vial"
    },
) as dag:

    conectar_minio = PythonOperator(
        task_id="conectar_minio",
        python_callable=ejemplo_conexion_s3
    )

    
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
    
    listar_boletines_task = PythonOperator(
        task_id="listar_pdfs_minio",
        python_callable=listar_pdfs_minio,
        provide_context=True,
        params={
            "bucket_name": "respaldo2",
            "prefix": "boletines/2025/",          # el mismo prefix que usaste al subir
            "aws_conn_id": "minio_s3",
            #"manifest_key": "boletines/2025/_manifest.json",
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

    build_bm25 = PythonOperator(
        task_id="build_bm25",
        python_callable=task_build_bm25_from_ndjson,
        op_kwargs={
            "bucket_name": "respaldo2",
            "prefix_chunks": "rag/chunks/2025/",
            "prefix_models": "rag/models/2025/",
            "aws_conn_id": "minio_s3",
        },
    )


    query_bm25 = PythonOperator(
        task_id="query_bm25_demo",
        python_callable=task_query_bm25,
        op_kwargs={
            "bucket_name": "respaldo2",
            "model_key":   "rag/models/2025/bm25.pkl",
            "aws_conn_id": "minio_s3",
            "query":       "contratación pública vial",  # <-- poné tu consulta de prueba
            "top_k":       5,
            "prefix_pdfs": "boletines/2025/",
        },
    )

    make_qrels = PythonOperator(
        task_id="make_qrels",
        python_callable=task_make_qrels_from_bm25,
        op_kwargs={
            "bucket_name": "respaldo2",
            "model_key":   "rag/models/2025/bm25.pkl",
            "aws_conn_id": "minio_s3",
            "qrels_key":   "rag/qrels/2025/qrels.csv",   # <-- donde lo espera eval_bm25
            "query":       "contratación pública vial",   # usa la misma que en query_bm25_demo
            "top_k_pos":   10,
            "add_negatives": True,
            "negatives_from_chunks_prefix": "rag/chunks/2025/",
            "negatives_count": 30,
        },
    )

    eval_bm25 = PythonOperator(
        task_id="eval_bm25",
        python_callable=task_eval_bm25,
        op_kwargs={
            "bucket_name": "respaldo2",
            "model_key":   "rag/models/2025/bm25.pkl",
            "qrels_key":   "rag/qrels/2025/qrels.csv",   # súbilo a ese path en MinIO
            "aws_conn_id": "minio_s3",
            "prefix_eval": "rag/eval/2025/",
            "k_list":      [5, 10],
            "top_k_search": 50,
        },
    )
    
    pinecone_upsert = PythonOperator(
        task_id="pinecone_upsert",
        python_callable=task_pinecone_upsert,
        op_kwargs={
            "bucket_name":  "respaldo2",
            "prefix_chunks":"rag/chunks/2025/",
            "aws_conn_id":  "minio_s3",
            "index_name":   "boletines-2025",
            "namespace":    "2025",
            "model_name":   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "batch_size":   128,
        },
    )

    pinecone_query = PythonOperator(
        task_id="pinecone_query_demo",
        python_callable=task_pinecone_query,
        op_kwargs={
            "index_name": "boletines-2025",
            "namespace":  "2025",
            "query":      "{{ params.eval_query }}",  # reutilizá tu param del DAG
            "top_k":      5,
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    )


    fusion_query = PythonOperator(
        task_id="fusion_query_demo",
        python_callable=task_fusion_query,
        op_kwargs={
            "bucket_name":  "respaldo2",
            "aws_conn_id":  "minio_s3",
            "bm25_model_key": "rag/models/2025/bm25.pkl",
            "top_k_bm25":     50,
            "pc_index_name":  "boletines-2025",
            "pc_namespace":   "2025",
            "top_k_vec":      50,
            "model_name":     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "rrf_k":          60,
            "top_k_fused":    10,
            "prefix_chunks":  "rag/chunks/2025/",
            "query":          "{{ params.eval_query }}",   # reutilizamos tu param
            "out_prefix":     "rag/fusion/2025/",
        },
    )
    


    eval_fusion = PythonOperator(
        task_id="eval_fusion",
        python_callable=task_eval_fusion,
        op_kwargs={
            "bucket_name":   "respaldo2",
            "aws_conn_id":   "minio_s3",
            "fusion_prefix": "rag/fusion/2025/",
            "qrels_key":     "rag/qrels/2025/qrels.csv",
            "metrics_prefix":"rag/metrics/2025/",
            "query":         "{{ params.eval_query }}",   # opcional; si no, toma el último fusion_*.json
            "ks":            [1, 3, 5, 10],
        },
    )



    # dump_docids = PythonOperator(
    #     task_id="dump_bm25_docids",
    #     python_callable=task_dump_doc_ids,
    #     op_kwargs={
    #         "bucket_name": "respaldo2",
    #         "model_key":   "rag/models/2025/bm25.pkl",
    #         "out_key":     "rag/qrels/2025/_docids.txt",
    #         "aws_conn_id": "minio_s3",
    #     },
    # )






    # Definimos el flujo de dependencias
    conectar_minio >> descargar_boletines_task >> listar_boletines_task >> extract_texts >> chunk_from_txt >> build_bm25 >> query_bm25 >> make_qrels >> eval_bm25
    
    chunk_from_txt >> pinecone_upsert >> pinecone_query
    
    # Fusión cuando ambos están listos:
    [query_bm25, pinecone_query] >> fusion_query
    # ahora añadí:
    fusion_query >> eval_fusion

