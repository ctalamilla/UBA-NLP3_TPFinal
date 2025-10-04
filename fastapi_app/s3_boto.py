# fastapi_app/s3_boto.py
import os, json
import boto3
from botocore.config import Config
from typing import List, Optional

def build_s3():
    endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT_URL") or "http://minio:9000"
    ak = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ACCESS_KEY") or "minio_admin"
    sk = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_SECRET_KEY") or "minio_admin"
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )

def list_keys(bucket: str, prefix: str, suffix: Optional[str] = None) -> List[str]:
    s3 = build_s3()
    keys: List[str] = []
    cont = None
    while True:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=cont) if cont else \
               s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if not suffix or k.lower().endswith(suffix.lower()):
                keys.append(k)
        if resp.get("IsTruncated"):
            cont = resp.get("NextContinuationToken")
        else:
            break
    return keys

def read_text(bucket: str, key: str, encoding: str = "utf-8") -> str:
    s3 = build_s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode(encoding)

def upload_json(bucket: str, key: str, obj) -> None:
    s3 = build_s3()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
