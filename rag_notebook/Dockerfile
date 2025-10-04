FROM jupyter/minimal-notebook:python-3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/home/jovyan/work:/home/jovyan/work/tasks:/home/jovyan/work/fastapi_app:$PYTHONPATH
