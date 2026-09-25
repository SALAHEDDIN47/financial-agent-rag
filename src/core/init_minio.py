# src/core/init_minio.py
"""
Crée les buckets MinIO nécessaires au projet.
Idempotent : peut être relancé sans risque.

Usage (dans n'importe quel conteneur ayant accès à src/) :
    python -m src.core.init_minio
"""
import sys

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.config.settings import settings

BUCKETS = [
    "raw-documents",
    "parsed-documents",
    "processed-chunks",
    "manifests",
]


def main() -> int:
    print(f"🔌 Connexion à MinIO : {settings.minio_endpoint}")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    try:
        existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    except ClientError as e:
        print(f"❌ Impossible de contacter MinIO : {e}")
        return 1

    print(f"📦 Buckets existants : {sorted(existing) or 'aucun'}")

    for bucket in BUCKETS:
        if bucket in existing:
            print(f"  ⏭️  {bucket} existe déjà")
            continue
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"  ✅ {bucket} créé")
        except ClientError as e:
            print(f"  ❌ Échec création {bucket} : {e}")

    print("\n✅ Tous les buckets sont prêts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())