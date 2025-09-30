FROM apache/airflow:2.7.1

USER root

RUN apt-get update && apt-get install -y libgomp1

# Crear carpeta de artefactos
#RUN mkdir -p /opt//mlflow/artifacts

# Asignar permisos solo al usuario airflow
# RUN chown -R airflow /mlflow/artifacts
# RUN chmod -R 777 /mlflow/artifacts

USER airflow