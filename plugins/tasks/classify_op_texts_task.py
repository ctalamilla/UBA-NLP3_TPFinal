# plugins/tasks/classify_op_texts_task.py
from __future__ import annotations
import os
import re
import json
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, List

# ---------- OpenAI (SDK v1.x) opcional ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # por si el paquete no está

# ---------- Preferimos tus utilidades S3 si existen ----------
try:
    from tasks.s3_utils import read_text as _read_text        # (bucket, key, aws_conn_id) -> str
    from tasks.s3_utils import upload_text as _upload_text    # (bucket, key, text, aws_conn_id) -> None
    HAVE_HELPERS = True
except Exception:
    HAVE_HELPERS = False

import boto3
from botocore.config import Config

# ==========================================================
# CATEGORÍAS (canónicas)
# ==========================================================
CANON_CATEGORIES = [
    'ADJUDICACIONES SIMPLES',
    'ASAMBLEAS CIVILES',
    'ASAMBLEAS COMERCIALES',
    'ASAMBLEAS PROFESIONALES',
    'AVISOS ADMINISTRATIVOS',
    'AVISOS COMERCIALES',
    'AVISOS GENERALES',
    'CITACIONES ADMINISTRATIVAS',
    'CONCESIONES DE AGUA PÚBLICA',
    'CONCURSOS CIVILES O PREVENTIVOS',
    'CONSTITUCIONES DE SOCIEDAD',
    'CONTRATACIONES ABREVIADAS',
    'CONVOCATORIAS A AUDIENCIA PÚBLICA',
    'CONVOCATORIAS A ELECCIONES',
    'COSAYSA -',
    'DECISIONES ADMINISTRATIVAS',
    'DECRETOS',
    'DETERMINACIÓN DE LÍNEA DE RIBERA',
    'DISPOSICIONES',
    'EDICTOS DE CATEOS',
    'EDICTOS DE MENSURAS',
    'EDICTOS DE MINAS',
    'EDICTOS DE QUIEBRAS',
    'EDICTOS JUDICIALES',
    'EL MINISTRO DE SALUD PÚBLICA',
    'ENTE REGULADOR DE LOS SERVICIOS PÚBLICOS',
    'ESTADOS CONTABLES',
    'FE DE ERRATAS',
    'LA MINISTRA DE EDUCACIÓN, CULTURA, CIENCIA Y TECNOLOGÍA',
    'LA SECRETARIA DE MINERÍA Y ENERGÍA DE LA PROVINCIA DE SALTA',
    'LEYES',
    'LICITACIONES PÚBLICAS',
    'NOTIFICACIONES ADMINISTRATIVAS',
    'POSESIONES VEINTEAÑALES',
    'RECAUDACIÓN',
    'REMATES ADMINISTRATIVOS',
    'REMATES JUDICIALES',
    'RESOLUCIONES',
    'RESOLUCIONES DE OTROS ORGANISMOS',
    'RESOLUCIONES DE OTROS ORGANISMOS CONJUNTAS',
    'RESOLUCIONES DE OTROS ORGANISMOS SINTETIZADAS',
    'RESOLUCIONES DELEGADAS',
    'RESOLUCIONES MINISTERIALES',
    'RESOLUCIONES SINTETIZADAS',
    'SECRETARIA GENERAL DE LA GOBERNACIÓN',
    'SECRETARÍA GENERAL DE LA GOBERNACIÓN',
    'SENTENCIAS',
    'SUCESORIOS',
    'TRANSFERENCIAS DE FONDO DE COMERCIO',
]

# ==========================================================
# Utilidades
# ==========================================================
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower()

def _re(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE | re.DOTALL)

# Palabras clave útiles
RX = {
    "edicto": _re(r"\bedicto"),
    "quiebra": _re(r"\bquiebra"),
    "mensura": _re(r"\bmensura"),
    "cateo": _re(r"\bcateo[s]?\b"),
    "mina": _re(r"\bmina[s]?\b"),
    "concurso": _re(r"\bconcurso[s]?\b|\bpreventivo\b"),
    "sentencia": _re(r"\bsentencia[s]?\b|juzgado|juez|secretaria"),
    "remate": _re(r"\bremate[s]?\b|subasta[s]?"),
    "judicial": _re(r"\bjudicial\b|juzgado|juez"),
    "administrativo": _re(r"\badministrativ[oa]s?\b|administracion"),
    "licitacion": _re(r"\blicitacion(?:es)?\b"),
    "publica": _re(r"\bpublica[s]?\b"),
    "adjudicacion_simple": _re(r"\badjudicacion(?:es)? simple[s]?\b"),
    "contratacion_abreviada": _re(r"\bcontratacion(?:es)? abreviada[s]?\b"),
    "concesion_agua": _re(r"\bconcesion(?:es)? de agua publica\b"),
    "ley": _re(r"\bley\s*n[°º]?\s*\d+"),
    "decreto": _re(r"\bdecreto[s]?\s*n[°º]?\s*\d+|\bdecreta"),
    "resolucion": _re(r"\bresolucion(?:es)?\b"),
    "res_min": _re(r"\bresolucion(?:es)? ministerial(?:es)?\b"),
    "res_delegada": _re(r"\bresolucion(?:es)? delegad(?:a|as)\b"),
    "res_otros": _re(r"\bresolucion(?:es)? de otros organismos\b"),
    "res_otros_conj": _re(r"\bresolucion(?:es)? de otros organismos conjuntas?\b"),
    "res_otros_sint": _re(r"\bresolucion(?:es)? de otros organismos sintetizada[s]?\b"),
    "res_sint": _re(r"\bresolucion(?:es)? sintetizada[s]?\b"),
    "disposicion": _re(r"\bdisposicion(?:es)?\b"),
    "decision_admin": _re(r"\bdecision(?:es)? administrativa[s]?\b"),
    "fe_erratas": _re(r"\bfe de erratas\b"),
    "conv_aud_pub": _re(r"\bconvocatoria[s]? a audiencia publica[s]?\b"),
    "conv_elecciones": _re(r"\bconvocatoria[s]? a elecciones\b"),
    "asamblea": _re(r"\basamblea[s]?\b"),
    "sociedad_comercial": _re(r"\bs\.?a\.?\b|\bs\.?r\.?l\.?\b|\bs\.?a\.?s\.?\b|\bsociedad\b|\bsa\b|\bsrl\b|\bsas\b"),
    "colegio_prof": _re(r"\bcolegio de\b|\bconsejo profesional\b|\bmatricula\b|\borden\b"),
    "asociacion_civil": _re(r"\basociacion civil\b|\bfundacion\b|\bclub\b|\bmutual\b|\bcooperadora\b"),
    "constitucion_soc": _re(r"\bconstitucion(?:es)? de sociedad(?:es)?\b|\bacta constitutiva\b"),
    "notificacion_admin": _re(r"\bnotificacion(?:es)? administrativa[s]?\b"),
    "citacion_admin": _re(r"\bcitacion(?:es)? administrativa[s]?\b"),
    "aviso_admin": _re(r"\baviso[s]? administrativo[s]?\b"),
    "aviso_com": _re(r"\baviso[s]? comercial(?:es)?\b"),
    "aviso_gral": _re(r"\baviso[s]? general(?:es)?\b"),
    "linea_ribera": _re(r"\blinea de ribera\b"),
    "posesion_veinte": _re(r"\bposesion(?:es)? veintean(?:al|ales)\b"),
    "transfer_fondo": _re(r"\btransferencia[s]? de fondo de comercio\b"),
    "estados_contables": _re(r"\bestados? contable[s]?\b|balance|memoria|cuentas?"),
    "recaudacion": _re(r"\brecaudacion\b|impuesto[s]?|tasas?"),
    "ente_regulador": _re(r"\bente regulador de los servicios publicos\b"),
    "cosaysa": _re(r"\bco\.?sa\.?y\.?sa\b|\baguas del norte\b"),
    "sg_gob": _re(r"\bsecretaria general de la gobernacion\b"),
    "min_salud": _re(r"\bel ministro de salud publica\b"),
    "min_educ": _re(r"\bla ministra de educacion, cultura, ciencia y tecnologia\b"),
    "min_mineria": _re(r"\bla secretaria de mineria y energia\b"),
    "edictos": _re(r"\bedicto[s]?\b"),
    "judicial_terms": _re(r"\bjuzgado|secretaria|expte|expediente|juez|tribunal\b"),
    "minas_terms": _re(r"\bmina[s]?|cateo[s]?|mensura[s]?\b"),
}

def _has(rx_key: str, txt: str) -> bool:
    return bool(RX[rx_key].search(txt))

def _is_asamblea_civil(txt: str) -> bool:
    return _has("asamblea", txt) and _has("asociacion_civil", txt)

def _is_asamblea_comercial(txt: str) -> bool:
    return _has("asamblea", txt) and _has("sociedad_comercial", txt)

def _is_asamblea_profesional(txt: str) -> bool:
    return _has("asamblea", txt) and _has("colegio_prof", txt)

def heuristic_category(text: str) -> str:
    t = _norm(text)

    # --- EDICTOS ESPECÍFICOS ---
    if _has("edictos", t) and _has("quiebra", t):
        return "EDICTOS DE QUIEBRAS"
    if _has("edictos", t) and _has("mensura", t):
        return "EDICTOS DE MENSURAS"
    if _has("edictos", t) and _has("cateo", t):
        return "EDICTOS DE CATEOS"
    if _has("edictos", t) and _has("mina", t):
        return "EDICTOS DE MINAS"
    if _has("edictos", t) and (_has("judicial_terms", t) or _has("judicial", t)):
        return "EDICTOS JUDICIALES"

    # --- REMATES ---
    if _has("remate", t) and _has("judicial", t):
        return "REMATES JUDICIALES"
    if _has("remate", t):
        return "REMATES ADMINISTRATIVOS"

    # --- CONCURSOS / QUIEBRA ---
    if _has("concurso", t):
        return "CONCURSOS CIVILES O PREVENTIVOS"

    # --- ASAMBLEAS ---
    if _is_asamblea_comercial(t):
        return "ASAMBLEAS COMERCIALES"
    if _is_asamblea_profesional(t):
        return "ASAMBLEAS PROFESIONALES"
    if _is_asamblea_civil(t):
        return "ASAMBLEAS CIVILES"

    # --- LICITACIONES / ADJUDICACIONES / CONTRATACIONES ---
    if _has("licitacion", t) and _has("publica", t):
        return "LICITACIONES PÚBLICAS"
    if _has("adjudicacion_simple", t):
        return "ADJUDICACIONES SIMPLES"
    if _has("contratacion_abreviada", t):
        return "CONTRATACIONES ABREVIADAS"

    # --- SECCIONES ESPECÍFICAS ---
    if _has("cosaysa", t):
        return "COSAYSA -"
    if _has("ente_regulador", t):
        return "ENTE REGULADOR DE LOS SERVICIOS PÚBLICOS"
    if _has("sg_gob", t):
        return "SECRETARÍA GENERAL DE LA GOBERNACIÓN"
    if _has("min_salud", t):
        return "EL MINISTRO DE SALUD PÚBLICA"
    if _has("min_educ", t):
        return "LA MINISTRA DE EDUCACIÓN, CULTURA, CIENCIA Y TECNOLOGÍA"
    if _has("min_mineria", t):
        return "LA SECRETARIA DE MINERÍA Y ENERGÍA DE LA PROVINCIA DE SALTA"

    # --- NORMATIVA ---
    if _has("ley", t):
        return "LEYES"
    if _has("decreto", t):
        return "DECRETOS"
    if _has("res_min", t):
        return "RESOLUCIONES MINISTERIALES"
    if _has("res_delegada", t):
        return "RESOLUCIONES DELEGADAS"
    if _has("res_otros_conj", t):
        return "RESOLUCIONES DE OTROS ORGANISMOS CONJUNTAS"
    if _has("res_otros_sint", t):
        return "RESOLUCIONES DE OTROS ORGANISMOS SINTETIZADAS"
    if _has("res_otros", t):
        return "RESOLUCIONES DE OTROS ORGANISMOS"
    if _has("res_sint", t):
        return "RESOLUCIONES SINTETIZADAS"
    if _has("resolucion", t):
        return "RESOLUCIONES"
    if _has("disposicion", t):
        return "DISPOSICIONES"
    if _has("decision_admin", t):
        return "DECISIONES ADMINISTRATIVAS"

    # --- AVISOS / NOTIFICACIONES / CITACIONES ---
    if _has("notificacion_admin", t):
        return "NOTIFICACIONES ADMINISTRATIVAS"
    if _has("citacion_admin", t):
        return "CITACIONES ADMINISTRATIVAS"
    if _has("aviso_admin", t):
        return "AVISOS ADMINISTRATIVOS"
    if _has("aviso_com", t):
        return "AVISOS COMERCIALES"
    if _has("aviso_gral", t):
        return "AVISOS GENERALES"

    # --- OTRAS ESPECÍFICAS ---
    if _has("linea_ribera", t):
        return "DETERMINACIÓN DE LÍNEA DE RIBERA"
    if _has("posesion_veinte", t):
        return "POSESIONES VEINTEAÑALES"
    if _has("transfer_fondo", t):
        return "TRANSFERENCIAS DE FONDO DE COMERCIO"
    if _has("concesion_agua", t):
        return "CONCESIONES DE AGUA PÚBLICA"
    if _has("estados_contables", t):
        return "ESTADOS CONTABLES"
    if _has("fe_erratas", t):
        return "FE DE ERRATAS"
    if _has("recaudacion", t):
        return "RECAUDACIÓN"
    if _has("sentencia", t):
        return "SENTENCIAS"
    if re.compile(r"\bsucesorio[s]?\b", re.IGNORECASE).search(t):
        return "SUCESORIOS"
    if re.compile(r"\bconstitucion(?:es)? de sociedad(?:es)?\b|\bacta constitutiva\b", re.IGNORECASE).search(t):
        return "CONSTITUCIONES DE SOCIEDAD"
    if re.compile(r"\bconvocatoria[s]? a audiencia publica", re.IGNORECASE).search(t):
        return "CONVOCATORIAS A AUDIENCIA PÚBLICA"
    if re.compile(r"\bconvocatoria[s]? a elecciones", re.IGNORECASE).search(t):
        return "CONVOCATORIAS A ELECCIONES"

    return "AVISOS GENERALES"

# ==========================================================
# OpenAI helpers (opcional)
# ==========================================================
def _make_client_from_env() -> Optional[OpenAI]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def llm_extract_fields(text: str, client: OpenAI, model: Optional[str] = None) -> dict:
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    snippet = (text or "")[:1600]
    prompt = f"""
Devuelve SOLO un JSON válido con los campos:
- numero
- fecha
- organismo
- personas (lista)
- resumen (máx 280 chars)

Texto:
{snippet}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if not isinstance(data.get("personas", []), list):
            data["personas"] = []
        data["resumen"] = (data.get("resumen") or "")[:280]
        return data
    except Exception:
        return {
            "numero": None,
            "fecha": None,
            "organismo": None,
            "personas": [],
            "resumen": (snippet or "")[:280],
        }

# ==========================================================
# S3 helpers
# ==========================================================
def _boto_client():
    endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT_URL") or "http://minio:9000"
    ak = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_admin")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )

def s3_read_text(bucket: str, key: str, aws_conn_id: Optional[str] = None) -> str:
    if HAVE_HELPERS:
        return _read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
    s3 = _boto_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="replace")

def s3_upload_text(bucket: str, key: str, text: str, aws_conn_id: Optional[str] = None) -> None:
    if HAVE_HELPERS:
        return _upload_text(bucket=bucket, key=key, text=text, aws_conn_id=aws_conn_id)
    s3 = _boto_client()
    s3.put_object(Bucket=bucket, Key=key, Body=(text or "").encode("utf-8"))

def s3_exists(bucket: str, key: str) -> bool:
    s3 = _boto_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def s3_list_keys(bucket: str, prefix: str, suffix: Optional[str] = None) -> List[str]:
    s3 = _boto_client()
    keys: List[str] = []
    token = None
    while True:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token) if token else \
               s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if not suffix or k.lower().endswith(suffix.lower()):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

# ==========================================================
# JSONL helpers
# ==========================================================
def _read_jsonl(bucket: str, key: str, aws_conn_id: Optional[str]) -> List[Dict[str, Any]]:
    raw = s3_read_text(bucket, key, aws_conn_id)
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def _write_jsonl(bucket: str, key: str, items: List[Dict[str, Any]], aws_conn_id: Optional[str]) -> None:
    body = "\n".join(json.dumps(it, ensure_ascii=False) for it in items)
    s3_upload_text(bucket, key, body, aws_conn_id)

def _default_sidecar_key(txt_key: str, sidecar_prefix: Optional[str]) -> Optional[str]:
    if not sidecar_prefix:
        return None
    base = os.path.basename(txt_key)
    return f"{sidecar_prefix.rstrip('/')}/{base.replace('.txt', '.meta.json')}"

# ==========================================================
# TASK principal
# ==========================================================
def task_classify_op_texts(
    bucket_name: str,
    meta_key_in: Optional[str] = None,
    meta_key_out: Optional[str] = None,
    aws_conn_id: Optional[str] = None,
    # NUEVO: modo “update sidecars” directamente en rag/text_op_meta/2025/
    meta_prefix: Optional[str] = None,    # ej: "rag/text_op_meta/2025/"
    force_overwrite: bool = False,        # True = regraba aunque ya exista classification
    # Opcionales de clasificación
    use_llm: bool = True,
    model_name: Optional[str] = None,
    write_sidecars: bool = False,
    sidecar_prefix: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Dos modos:
    A) Índice JSONL -> JSONL (compatibilidad hacia atrás): si se pasa meta_key_in (y existe),
       lee cada item, clasifica y escribe meta_key_out (y opcionalmente sidecars).
    B) Actualización in-place de sidecars existentes: si NO se pasa meta_key_in,
       recorre meta_prefix/docs/*.meta.json, lee el TXT referido en cada sidecar, clasifica y
       sobreescribe el MISMO sidecar agregando 'classification'. Además genera _index_classified.jsonl.
    """
    now_iso = datetime.utcnow().isoformat() + "Z"
    client = _make_client_from_env() if use_llm else None
    if client is None and use_llm:
        print("⚠️ OPENAI_API_KEY no configurada o SDK no disponible. Se usará SOLO heurística.")

    # -----------------------------
    # MODO A: JSONL -> JSONL
    # -----------------------------
    if meta_key_in:
        if not s3_exists(bucket_name, meta_key_in):
            raise FileNotFoundError(f"No existe s3://{bucket_name}/{meta_key_in}")

        records = _read_jsonl(bucket_name, meta_key_in, aws_conn_id)
        if not records:
            print(f"ℹ️ Índice vacío: s3://{bucket_name}/{meta_key_in}")
            if meta_key_out:
                _write_jsonl(bucket_name, meta_key_out, [], aws_conn_id)
            return {
                "mode": "index_jsonl",
                "bucket": bucket_name,
                "meta_in": meta_key_in,
                "meta_out": meta_key_out,
                "processed": 0,
                "classified": 0,
                "sidecars": 0,
            }

        out_items: List[Dict[str, Any]] = []
        wrote_sidecars = 0
        classified = 0

        for it in records:
            txt_key = it.get("txt_key")
            if not txt_key:
                out_items.append(it)
                continue

            try:
                text = s3_read_text(bucket_name, txt_key, aws_conn_id)
            except Exception as e:
                print(f"⚠️ No se pudo leer TXT '{txt_key}': {e}")
                out_items.append(it)
                continue

            # Saltar si ya tiene clasificación y no queremos sobreescribir
            if not force_overwrite and it.get("classification"):
                out_items.append(it)
                continue

            cat = heuristic_category(text)
            origin = "heuristic"
            extracted = None
            if client is not None:
                extracted = llm_extract_fields(text, client=client, model=model_name)
                origin = "heuristic+llm_meta"

            classification = {
                "categoria": cat,
                "origin": origin,
                "at": now_iso,
            }
            if extracted:
                classification["extracted"] = extracted

            new_item = dict(it)
            new_item["classification"] = classification
            out_items.append(new_item)
            classified += 1

            if write_sidecars:
                skey = _default_sidecar_key(txt_key, sidecar_prefix)
                if skey:
                    try:
                        s3_upload_text(bucket_name, skey, json.dumps(new_item, ensure_ascii=False, indent=2), aws_conn_id)
                        wrote_sidecars += 1
                    except Exception as e:
                        print(f"⚠️ Error escribiendo sidecar '{skey}': {e}")

        # meta_key_out por defecto
        if not meta_key_out:
            if meta_key_in.endswith("_index.jsonl"):
                meta_key_out = meta_key_in.replace("_index.jsonl", "_index_classified.jsonl")
            else:
                base, ext = os.path.splitext(meta_key_in)
                meta_key_out = f"{base}_classified{ext or '.jsonl'}"

        _write_jsonl(bucket_name, meta_key_out, out_items, aws_conn_id)
        print(f"✅ Clasificados {classified} items → s3://{bucket_name}/{meta_key_out}")

        return {
            "mode": "index_jsonl",
            "bucket": bucket_name,
            "meta_in": meta_key_in,
            "meta_out": meta_key_out,
            "processed": len(records),
            "classified": classified,
            "sidecars": wrote_sidecars,
            "sample": out_items[:5],
        }

    # -----------------------------
    # MODO B: Actualizar sidecars existentes en meta_prefix
    # -----------------------------
    meta_prefix = meta_prefix or "rag/text_op_meta/2025/"
    docs_prefix = f"{meta_prefix.rstrip('/')}/"
    sidecars = s3_list_keys(bucket_name, docs_prefix, suffix=".meta.json")
    if not sidecars:
        raise FileNotFoundError(
            f"No se encontraron sidecars en s3://{bucket_name}/{docs_prefix} "
            f"(asegurate de haber corrido la tarea de extracción que generó metadatos)"
        )

    out_items: List[Dict[str, Any]] = []
    classified = 0

    for skey in sidecars:
        try:
            item_raw = s3_read_text(bucket_name, skey, aws_conn_id)
            item = json.loads(item_raw)
        except Exception as e:
            print(f"⚠️ No se pudo leer o parsear sidecar '{skey}': {e}")
            continue

        txt_key = item.get("txt_key")
        if not txt_key:
            print(f"⚠️ Sidecar sin 'txt_key': {skey}")
            continue

        # Saltar si ya tiene clasificación y no queremos sobreescribir
        if item.get("classification") and not force_overwrite:
            out_items.append(item)
            continue

        try:
            text = s3_read_text(bucket_name, txt_key, aws_conn_id)
        except Exception as e:
            print(f"⚠️ No se pudo leer TXT '{txt_key}': {e}")
            out_items.append(item)
            continue

        cat = heuristic_category(text)
        origin = "heuristic"
        extracted = None
        if use_llm:
            client = client or _make_client_from_env()
            if client is not None:
                extracted = llm_extract_fields(text, client=client, model=model_name)
                origin = "heuristic+llm_meta"

        classification = {
            "categoria": cat,
            "origin": origin,
            "at": now_iso,
        }
        if extracted:
            classification["extracted"] = extracted

        item["classification"] = classification
        # persistir sobre el MISMO sidecar
        try:
            s3_upload_text(bucket_name, skey, json.dumps(item, ensure_ascii=False, indent=2), aws_conn_id)
            classified += 1
        except Exception as e:
            print(f"⚠️ Error escribiendo sidecar '{skey}': {e}")

        out_items.append(item)

    # También escribimos un índice clasificado para consumo rápido
    classified_index_key = f"{meta_prefix.rstrip('/')}/_index_classified.jsonl"
    _write_jsonl(bucket_name, classified_index_key, out_items, aws_conn_id)
    print(f"✅ Actualizados {classified} sidecars y generado índice → s3://{bucket_name}/{classified_index_key}")

    return {
        "mode": "update_sidecars",
        "bucket": bucket_name,
        "meta_prefix": meta_prefix,
        "docs_prefix": docs_prefix,
        "index_classified": classified_index_key,
        "processed": len(sidecars),
        "classified": classified,
        "sample": out_items[:5],
    }
