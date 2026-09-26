# src/indexing/embedder.py
"""
Embedder S3-aware.

Lit   : processed-chunks/{company}/{doc_type}/{source}_chunks.jsonl
Écrit : embeddings/{company}/{doc_type}/{source}_chunks.jsonl  (avec field "embedding")
Manifest : manifests/embedding_manifest.json
"""
import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.storage import storage, compute_key_hash

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_KEY = "manifests/embedding_manifest.json"
INPUT_PREFIX = "processed-chunks/"
OUTPUT_PREFIX = "embeddings/"


def _detect_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU : {torch.cuda.get_device_name(0)}")
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


class ChunkEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = _detect_device(device)
        self.batch_size = batch_size
        logger.info(f"🧠 Chargement {model_name} sur {self.device}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✅ Modèle chargé en {time.time()-t0:.1f}s")
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"📐 Dim = {self.dim}")

    def embed_texts(self, jsonl_text: str) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []
        for line in jsonl_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                import json
                chunk = json.loads(line)
            except Exception:
                continue
            if len(chunk.get("text", "").strip()) < 10:
                continue
            chunks.append(chunk)

        if not chunks:
            return {"chunks": 0, "jsonl_out": ""}

        texts = [c["text"] for c in chunks]
        t0 = time.time()
        embs = self.model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )
        import json
        lines_out = []
        for chunk, emb in zip(chunks, embs):
            chunk["embedding"] = emb.tolist()
            meta = chunk.setdefault("metadata", {})
            meta["model_name"] = self.model_name
            meta["embedding_dim"] = self.dim
            lines_out.append(json.dumps(chunk, ensure_ascii=False))
        return {
            "chunks": len(chunks),
            "jsonl_out": "\n".join(lines_out),
            "duration_s": time.time() - t0,
        }


def embed_all(
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
    embedder: Optional[ChunkEmbedder] = None,
    force: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    embedder = embedder or ChunkEmbedder()
    manifest = storage.read_json(MANIFEST_KEY) if storage.exists(MANIFEST_KEY) else {}

    keys = [k for k in storage.list(input_prefix) if k.lower().endswith(".jsonl")]
    if limit:
        keys = keys[:limit]
    logger.info(f"🔍 {len(keys)} fichiers à embedder")

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_embedded": 0}

    for key in keys:
        prev = manifest.get(key)
        if not force and prev:
            try:
                current_hash = compute_key_hash(key)
            except Exception:
                current_hash = None
            if current_hash and current_hash == prev.get("source_hash"):
                out_key = prev.get("output_key")
                if out_key and storage.exists(out_key):
                    stats["skipped"] += 1
                    continue

        try:
            jsonl = storage.read_text(key)
            result = embedder.embed_texts(jsonl)
            if not result["jsonl_out"]:
                stats["failed"] += 1
                continue

            rel = key.replace(input_prefix, "")
            output_key = f"{output_prefix}{rel}"
            storage.write_text(output_key, result["jsonl_out"])

            manifest[key] = {
                "source_hash": compute_key_hash(key),
                "embedded_at": datetime.now(timezone.utc).isoformat(),
                "model_name": embedder.model_name,
                "dim": embedder.dim,
                "count": result["chunks"],
                "output_key": output_key,
            }
            stats["processed"] += 1
            stats["total_embedded"] += result["chunks"]
            logger.info(f"✅ {rel} : {result['chunks']} chunks")

            # Checkpoint toutes les 50
            if stats["processed"] % 50 == 0:
                storage.write_json(MANIFEST_KEY, manifest)
        except Exception as e:
            logger.error(f"❌ {key} : {e}")
            stats["failed"] += 1

    storage.write_json(MANIFEST_KEY, manifest)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_PREFIX)
    ap.add_argument("--output", default=OUTPUT_PREFIX)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Limiter pour test")
    args = ap.parse_args()

    embedder = ChunkEmbedder(batch_size=args.batch_size, device=args.device)
    t0 = time.time()
    stats = embed_all(args.input, args.output, embedder, args.force, args.limit)
    logger.info(f"\n🎉 Embedder terminé en {time.time()-t0:.1f}s")
    logger.info(f"   ✅ {stats['processed']} traités")
    logger.info(f"   ⏭️  {stats['skipped']} ignorés")
    logger.info(f"   ❌ {stats['failed']} échecs")
    logger.info(f"   📊 {stats['total_embedded']} chunks embeddés")