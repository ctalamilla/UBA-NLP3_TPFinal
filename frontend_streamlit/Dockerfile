FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias
COPY frontend_streamlit/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Código
COPY frontend_streamlit /app

# Vars por defecto (puedes sobreescribir en compose)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV RAG_API_URL=http://rag_api:8000

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
