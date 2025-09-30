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

# Definimos el DAG
with DAG(
    dag_id="flujo_RAG",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pipeline"]
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


    # Definimos el flujo de dependencias
    conectar_minio >> descargar_boletines_task >> listar_boletines_task >> extract_texts >> chunk_from_txt
