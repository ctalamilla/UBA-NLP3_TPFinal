FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/code"

WORKDIR /code

# Reqs
COPY fastapi_app/requirements.txt /code/fastapi_app/requirements.txt
RUN pip install --no-cache-dir -r /code/fastapi_app/requirements.txt

# Copiamos la app y los módulos 'tasks' usados por el pickle de BM25
COPY fastapi_app /code/fastapi_app
COPY plugins /code/plugins

# Exponer API
EXPOSE 8000

# Uvicorn
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
