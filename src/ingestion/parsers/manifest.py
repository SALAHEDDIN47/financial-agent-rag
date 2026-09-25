# src/ingestion/parsers/manifest.py
import logging

from src.core.storage import storage

logger = logging.getLogger(__name__)

MANIFEST_KEY = "manifests/parsing_manifest.json"


def load_manifest() -> dict:
    if storage.exists(MANIFEST_KEY):
        try:
            return storage.read_json(MANIFEST_KEY)
        except Exception as e:
            logger.warning(f"⚠️  Manifest corrompu ({e}), redémarrage à vide")
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    storage.write_json(MANIFEST_KEY, manifest)