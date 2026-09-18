# src/indexing/rebuild_embedding_manifest.py
"""
Reconstruit le manifest d'embedding depuis les fichiers déjà présents sur disque.

Utile quand :
  - Une ancienne version du script a tourné sans sauvegarder le manifest.
  - Le manifest a été perdu/corrompu mais les outputs existent.

Usage :
    uv run ./src/indexing/rebuild_embedding_manifest.py
    uv run ./src/indexing/rebuild_embedding_manifest.py --dry-run
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/processed/.embedding_manifest.json")
CHUNKS_DIR = Path("data/processed/chunks")
EMBEDDED_DIR = Path("data/processed/embedded_chunks")


def compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild(dry_run: bool = False) -> dict:
    # 1. Charger le manifest actuel
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        logger.info(f"📂 Manifest existant : {len(manifest)} entrées")
    else:
        manifest = {}
        logger.info("📂 Aucun manifest existant, création à partir de zéro")

    # 2. Parcourir les fichiers embeddés
    embedded_files = list(EMBEDDED_DIR.rglob("*.jsonl"))
    logger.info(f"🔍 {len(embedded_files)} fichiers embeddés trouvés sur disque")

    added = 0
    already_present = 0
    orphaned = 0  # embedded sans chunk source correspondant
    already_in_manifest = 0

    for emb_file in embedded_files:
        # Chemin relatif (préserve company/doc_type/)
        rel_path = emb_file.relative_to(EMBEDDED_DIR)

        # Fichier chunk source correspondant
        chunk_file = CHUNKS_DIR / rel_path
        if not chunk_file.exists():
            orphaned += 1
            continue

        key = str(chunk_file)
        if key in manifest:
            already_in_manifest += 1
            continue

        # Calculer le hash du chunk source
        try:
            source_hash = compute_hash(chunk_file)
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"⚠️  Impossible de hasher {chunk_file} : {e}")
            continue

        # Compter les chunks dans le fichier embeddé
        try:
            with open(emb_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
        except Exception:
            count = 0

        # Ajouter l'entrée au manifest
        manifest[key] = {
            "source_hash": source_hash,
            "embedded_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            "model_name": "BAAI/bge-large-en-v1.5",  # à ajuster si différent
            "dim": 1024,
            "count": count,
            "output": str(emb_file),
            "rebuilt_from_disk": True,   # marqueur : entrée reconstruite
        }
        added += 1

    # 3. Résumé
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 Résumé de la reconstruction")
    logger.info(f"   ✅ Entrées ajoutées       : {added}")
    logger.info(f"   ⏭️  Déjà dans le manifest  : {already_in_manifest}")
    logger.info(f"   ⚠️  Orphelines (sans chunk) : {orphaned}")
    logger.info(f"   📦 Manifest final          : {len(manifest)} entrées")
    logger.info(f"{'=' * 60}")

    # 4. Sauvegarder
    if dry_run:
        logger.warning("🧪 --dry-run : manifest NON sauvegardé")
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(MANIFEST_PATH)
        logger.info(f"💾 Manifest sauvegardé : {MANIFEST_PATH.absolute()}")

    return {
        "added": added,
        "already_in_manifest": already_in_manifest,
        "orphaned": orphaned,
        "total": len(manifest),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait fait sans modifier le manifest",
    )
    args = ap.parse_args()

    rebuild(dry_run=args.dry_run)