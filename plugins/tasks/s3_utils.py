# plugins/tasks/s3_utils.py
import os
import re
import time
from datetime import datetime, timedelta, timezone
import tempfile
import requests

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False
    
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from botocore.exceptions import ClientError

def ejemplo_conexion_s3():
    bucket_name = 'respaldo2'
    hook = S3Hook(aws_conn_id='minio_s3')
    s3_client = hook.get_conn()

    if not hook.check_for_bucket(bucket_name):
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"🪣 Bucket '{bucket_name}' creado correctamente.")
        except ClientError as e:
            print(f"❌ Error al crear bucket: {e}")
    else:
        print(f"✅ El bucket '{bucket_name}' ya existe.")

    s3_client.put_object(
        Bucket=bucket_name,
        Key='prueba.txt',
        Body='Desde Airflow por variable de entorno'
    )
    print(f"📄 Archivo 'prueba.txt' subido a bucket '{bucket_name}'.")

def descargar_dataset(**kwargs):
    url = 'https://docs.google.com/uc?export=download&id=1gT8k90Iisd-sZVXWtS6Exl1ZFwwTd_WM'
    nombre_archivo_local = 'dataset.csv'
    bucket_name = 'respaldo2'
    s3_key = 'dataset.csv'
    hook = S3Hook(aws_conn_id='minio_s3')

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = os.path.join(tmpdirname, nombre_archivo_local)
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)

        hook.load_file(
            filename=local_path,
            key=s3_key,
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Dataset subido a MinIO en {bucket_name}/{s3_key}")
        

def descargar_boletines_salta(**kwargs):
    """
    Descarga PDFs del Boletín Oficial de Salta (últimos N días) y los sube a MinIO.
    Parametrizable vía params del PythonOperator.
    """
    params = kwargs.get("params", {}) or {}

    # --- Parámetros con defaults razonables ---
    year = int(params.get("year", 2025))
    days = int(params.get("days", 14))  # ventana en días (2 semanas por defecto)
    url_listado = params.get(
        "url_listado",
        "https://boletinoficialsalta.gob.ar/Boletines_por_Anio.php?cXdlcnR5YW5pbz0yMDI1cXdlcnR5"
    )
    bucket_name = params.get("bucket_name", "respaldo2")
    prefix = params.get("prefix", f"boletines/{year}/")
    aws_conn_id = params.get("aws_conn_id", "minio_s3")
    timeout = int(params.get("timeout", 20))
    reintentos = int(params.get("reintentos", 3))
    pausa_retry = float(params.get("pausa_retry", 2.0))

    # --- Fecha de referencia en AR (UTC-3) ---
    hoy_ar = datetime.now(timezone(timedelta(hours=-3))).date()
    limite = hoy_ar - timedelta(days=days)

    hook = S3Hook(aws_conn_id=aws_conn_id)

    # --- Sesión HTTP ---
    ses = requests.Session()
    ses.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Airflow/requests scraper"
    })

    # --- Obtener y parsear listado ---
    resp = ses.get(url_listado, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    if _HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        texto = soup.get_text("\n", strip=True)
    else:
        # Fallback simple si no está bs4 (menos robusto)
        texto = re.sub(r"<[^>]+>", " ", html)

    # Busca pares: NÚMERO (5 dígitos) + FECHA (dd/mm/yyyy)
    patron = re.compile(r"\b(\d{5})\b\s+(\d{2}/\d{2}/\d{4})")
    hallazgos = patron.findall(texto)

    ediciones = []
    for numero_str, fecha_str in hallazgos:
        try:
            fecha = datetime.strptime(fecha_str, "%d/%m/%Y").date()
            numero = int(numero_str)
            ediciones.append((numero, fecha))
        except Exception:
            continue

    # Normaliza: únicos y orden
    ediciones = sorted(set(ediciones), key=lambda x: x[1])

    # Filtra últimas 'days'
    recientes = [(nro, fch) for (nro, fch) in ediciones if limite <= fch <= hoy_ar]

    if not recientes:
        print("ℹ️ No se encontraron boletines en la ventana solicitada.")
        return {"subidos": 0, "detalles": []}

    base_pdf = f"https://boletinoficialsalta.gob.ar/boletindigital/{year}"
    subidos = 0
    detalles = []

    def _descargar(url_pdf, destino_archivo):
        for intento in range(1, reintentos + 1):
            try:
                with ses.get(url_pdf, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(destino_archivo, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                return True
            except Exception as e:
                if intento == reintentos:
                    print(f"[WARN] Falló descarga {url_pdf}: {e}")
                    return False
                time.sleep(pausa_retry)

    with tempfile.TemporaryDirectory() as tmpdir:
        for nro, fch in recientes:
            nombre_local = f"{nro}_{fch.strftime('%Y-%m-%d')}.pdf"
            local_path = os.path.join(tmpdir, nombre_local)
            url_pdf = f"{base_pdf}/{nro}.pdf"
            s3_key = f"{prefix}{nombre_local}"

            ok = _descargar(url_pdf, local_path)
            if not ok:
                detalles.append({"nro": nro, "fecha": str(fch), "status": "descarga_fallida"})
                continue

            # Subir a MinIO
            try:
                hook.load_file(
                    filename=local_path,
                    key=s3_key,
                    bucket_name=bucket_name,
                    replace=True
                )
                subidos += 1
                detalles.append({"nro": nro, "fecha": str(fch), "status": f"subido:{bucket_name}/{s3_key}"})
                print(f"✅ PDF subido: s3://{bucket_name}/{s3_key}")
            except Exception as e:
                detalles.append({"nro": nro, "fecha": str(fch), "status": f"upload_error:{e}"})
                print(f"[ERROR] No se pudo subir {nombre_local}: {e}")

    print(f"✅ {subidos} boletines subidos a MinIO (bucket '{bucket_name}', prefix '{prefix}').")
    return {"subidos": subidos, "detalles": detalles}