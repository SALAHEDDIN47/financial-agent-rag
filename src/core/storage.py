# src/core/storage.py
"""
Abstraction du stockage objet (S3/MinIO) avec fallback local.

Deux backends :
  - "minio"  : production (lit/écrit dans MinIO)
  - "local"  : développement sans Docker (lit/écrit dans ./data)

Choix via variable d'env STORAGE_BACKEND (défaut : "minio").
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
from botocore.client import Config

from src.config.settings import settings


class Storage:
    """Interface unique pour lire/écrire des objets (S3 ou local)."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or settings.storage_backend).lower()
        if self.backend == "minio":
            self._init_s3()
        elif self.backend == "local":
            self._init_local()
        else:
            raise ValueError(f"Backend inconnu : {self.backend}")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_s3(self) -> None:
        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self._root = None  # pas utilisé pour S3

    def _init_local(self) -> None:
        self.s3 = None
        self._root = Path("data")
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        """'raw-documents/sec/AAPL/...' → ('raw-documents', 'sec/AAPL/...')"""
        key = key.lstrip("/")
        parts = key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Clé invalide (bucket manquant) : {key}")
        return parts[0], parts[1]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write_bytes(self, key: str, data: bytes) -> None:
        if self.backend == "minio":
            bucket, s3key = self._split_key(key)
            self.s3.put_object(Bucket=bucket, Key=s3key, Body=data)
        else:
            local = self._root / key
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def write_json(self, key: str, obj) -> None:
        self.write_text(key, json.dumps(obj, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read_bytes(self, key: str) -> bytes:
        if self.backend == "minio":
            bucket, s3key = self._split_key(key)
            resp = self.s3.get_object(Bucket=bucket, Key=s3key)
            return resp["Body"].read()
        local = self._root / key
        return local.read_bytes()

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    # ------------------------------------------------------------------
    # List / Exists / Delete
    # ------------------------------------------------------------------
    def exists(self, key: str) -> bool:
        if self.backend == "minio":
            bucket, s3key = self._split_key(key)
            try:
                self.s3.head_object(Bucket=bucket, Key=s3key)
                return True
            except self.s3.exceptions.ClientError:
                return False
        return (self._root / key).exists()

    def list(self, prefix: str) -> List[str]:
        """Retourne les clés complètes (bucket inclus) sous `prefix`."""
        prefix = prefix.lstrip("/")
        if self.backend == "minio":
            bucket, s3prefix = self._split_key(prefix)
            out = []
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=s3prefix):
                for obj in page.get("Contents", []):
                    out.append(f"{bucket}/{obj['Key']}")
            return out
        else:
            base = self._root / prefix
            if not base.exists():
                return []
            return [
                str(p.relative_to(self._root)).replace("\\", "/")
                for p in base.rglob("*") if p.is_file()
            ]

    def delete(self, key: str) -> None:
        if self.backend == "minio":
            bucket, s3key = self._split_key(key)
            self.s3.delete_object(Bucket=bucket, Key=s3key)
        else:
            (self._root / key).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Téléchargement temporaire (pour les parsers qui lisent un Path)
    # ------------------------------------------------------------------
    @contextmanager
    def open_local_temp(self, key: str) -> Iterator[Path]:
        """
        Télécharge l'objet dans un fichier temporaire et yield son Path.
        Nettoie automatiquement après usage.

        Utile pour les parsers existants qui attendent un `Path` en entrée.
        """
        suffix = Path(key).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tmp_path.write_bytes(self.read_bytes(key))
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)


# Singleton
storage = Storage()