# src/indexing/embedder.py
"""
Générateur d'embeddings pour les chunks financiers.

Fonctionnalités :
  - Détection automatique GPU → fallback CPU si CUDA indisponible
  - Batching (10-30x plus rapide que le 1-par-1)
  - Idempotence via manifest (data/processed/.embedding_manifest.json)
  - ✅ Checkpoints périodiques (tous les 100 fichiers OU 60 secondes)
  - ✅ Gestion propre de Ctrl+C (KeyboardInterrupt → sauvegarde du manifest)
  - Préserve la structure {company}/{doc_type}/ en sortie
  - Skip les chunks trop courts (< 10 caractères)

Usage :
    uv run ./src/indexing/embedder.py                     # traite les nouveaux
    uv run ./src/indexing/embedder.py --force             # re-embed tout
    uv run ./src/indexing/embedder.py --batch-size 16     # ajuster le batch
    uv run ./src/indexing/embedder.py --device cpu        # forcer CPU
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ==============================================================================
# DÉTECTION DU DEVICE (GPU → CPU fallback)
# ==============================================================================
def _detect_device(prefer: Optional[str] = None) -> str:
    """Détecte le meilleur device disponible : cuda > mps > cpu."""
    if prefer:
        return prefer

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU détecté : {gpu_name} ({vram:.1f} GB VRAM)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("🍎 Apple Silicon GPU (MPS) détecté")
        else:
            device = "cpu"
            logger.warning("⚠️  Aucun GPU détecté → fallback CPU (plus lent)")
    except ImportError:
        device = "cpu"
        logger.warning("⚠️  PyTorch non importable → fallback CPU")

    return device


# ==============================================================================
# MANIFEST D'IDEMPOTENCE
# ==============================================================================
MANIFEST_PATH = Path("data/processed/.embedding_manifest.json")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("⚠️  Manifest corrompu, redémarrage à vide")
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    """Sauvegarde atomique du manifest (évite la corruption si crash)."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Écriture atomique : .tmp puis rename
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(MANIFEST_PATH)


def _compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# EMBEDDER PRINCIPAL
# ==============================================================================
class ChunkEmbedder:
    """Génère les embeddings pour des fichiers JSONL de chunks."""

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

        logger.info(f"🧠 Chargement du modèle : {model_name} sur {self.device}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✅ Modèle chargé en {time.time() - t0:.1f}s")

        # ✅ Compatibilité avec anciennes et nouvelles versions de sentence-transformers
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"📐 Dimension d'embedding : {self.dim}")

    def embed_file(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Lit un JSONL, génère les embeddings par batch, écrit le JSONL enrichi."""
        # 1. Charger les chunks valides
        chunks: List[Dict[str, Any]] = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = chunk.get("text", "").strip()
                if len(text) < 10:
                    continue

                chunks.append(chunk)

        if not chunks:
            return {"chunks": 0, "embedded": 0, "skipped": 0, "duration_s": 0.0}

        # 2. Encoder par batchs
        t0 = time.time()
        texts = [c["text"] for c in chunks]
        all_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 3. Attacher les embeddings
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f_out:
            for chunk, emb in zip(chunks, all_embeddings):
                chunk["embedding"] = emb.tolist()
                meta = chunk.setdefault("metadata", {})
                meta["model_name"] = self.model_name
                meta["embedding_dim"] = self.dim
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        return {
            "chunks": len(texts),
            "embedded": len(all_embeddings),
            "skipped": 0,
            "duration_s": time.time() - t0,
        }


# ==============================================================================
# ORCHESTRATION (avec checkpoints périodiques + Ctrl+C propre)
# ==============================================================================
def embed_directory(
    input_dir: Path,
    output_dir: Path,
    embedder: ChunkEmbedder,
    force: bool = False,
    save_every: int = 100,
    save_interval_s: int = 60,
) -> Dict[str, Any]:
    """
    Parcourt récursivement les JSONL de chunks, génère les embeddings.

    Idempotence + robustesse :
      - Skip les fichiers déjà embeddés (hash identique).
      - ✅ Sauvegarde le manifest tous les `save_every` fichiers OU
        toutes les `save_interval_s` secondes.
      - ✅ Ctrl+C : sauvegarde du manifest avant de quitter.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()

    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    logger.info(f"🔍 {len(jsonl_files)} fichiers JSONL trouvés dans {input_dir}")

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0,
        "total_embedded": 0,
        "total_time_s": 0.0,
    }

    # Compteurs pour checkpoint
    processed_since_save = 0
    last_save_time = time.time()

    def _checkpoint(force_save: bool = False):
        """Sauvegarde le manifest si les seuils sont atteints."""
        nonlocal processed_since_save, last_save_time
        now = time.time()
        if (
            force_save
            or processed_since_save >= save_every
            or (now - last_save_time) >= save_interval_s
        ):
            _save_manifest(manifest)
            processed_since_save = 0
            last_save_time = now
            logger.debug(
                f"💾 Manifest sauvegardé ({len(manifest)} entrées)"
            )

    interrupted = False

    try:
        for input_file in jsonl_files:
            key = str(input_file)
            prev = manifest.get(key)

            # Chemin de sortie (préserve {company}/{doc_type}/)
            rel_path = input_file.relative_to(input_dir)
            output_file = output_dir / rel_path

            # Vérification idempotente
            if not force and prev and output_file.exists():
                try:
                    current_hash = _compute_hash(input_file)
                except (FileNotFoundError, PermissionError):
                    current_hash = None
                if current_hash and current_hash == prev.get("source_hash"):
                    stats["skipped"] += 1
                    # Pas de checkpoint sur les skips (rapides)
                    continue

            # Embedding effectif
            try:
                result = embedder.embed_file(input_file, output_file)

                manifest[key] = {
                    "source_hash": _compute_hash(input_file),
                    "embedded_at": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    "model_name": embedder.model_name,
                    "dim": embedder.dim,
                    "count": result["embedded"],
                    "output": str(output_file),
                }

                stats["processed"] += 1
                stats["total_chunks"] += result["chunks"]
                stats["total_embedded"] += result["embedded"]
                stats["total_time_s"] += result["duration_s"]
                processed_since_save += 1

                logger.info(
                    f"✅ {rel_path} : {result['embedded']} chunks "
                    f"en {result['duration_s']:.1f}s"
                )

                # ✅ Checkpoint si seuil atteint
                _checkpoint()

            except Exception as e:
                logger.error(f"❌ Échec : {rel_path} → {e}")
                stats["failed"] += 1

    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "\n⏸️  Ctrl+C détecté — sauvegarde du manifest avant arrêt..."
        )

    finally:
        # ✅ Sauvegarde finale (toujours, même en cas d'interruption)
        _save_manifest(manifest)
        logger.info(
            f"💾 Manifest sauvegardé : {len(manifest)} entrées au total"
        )

    if interrupted:
        logger.info(
            "\n💡 Pour reprendre là où vous vous êtes arrêté, relancez :"
        )
        logger.info("   uv run ./src/indexing/embedder.py")

    return stats


# ==============================================================================
# EXÉCUTION DIRECTE
# ==============================================================================
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-embed tout")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chunks"),
        help="Dossier des chunks JSONL",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/embedded_chunks"),
        help="Dossier de sortie",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Modèle SentenceTransformer",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda", "mps"],
        help="Device (None = auto-détection)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint tous les N fichiers traités",
    )
    ap.add_argument(
        "--save-interval",
        type=int,
        default=60,
        help="Checkpoint toutes les N secondes",
    )
    args = ap.parse_args()

    embedder = ChunkEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    t_global = time.time()
    stats = embed_directory(
        input_dir=args.input,
        output_dir=args.output,
        embedder=embedder,
        force=args.force,
        save_every=args.save_every,
        save_interval_s=args.save_interval,
    )
    elapsed = time.time() - t_global

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Embedding terminé !")
    logger.info(f"   ✅ Fichiers traités   : {stats['processed']}")
    logger.info(f"   ⏭️  Fichiers ignorés  : {stats['skipped']}")
    logger.info(f"   ❌ Échecs             : {stats['failed']}")
    logger.info(f"   📊 Chunks embeddés    : {stats['total_embedded']}")
    logger.info(f"   ⏱️  Temps d'embedding : {stats['total_time_s']:.1f}s")
    logger.info(f"   ⏱️  Temps total       : {elapsed:.1f}s")
    if stats["total_embedded"] > 0 and stats["total_time_s"] > 0:
        rate = stats["total_embedded"] / stats["total_time_s"]
        logger.info(f"   🚀 Débit moyen        : {rate:.1f} chunks/s")
    logger.info(f"   📁 Sortie             : {args.output.absolute()}")
    logger.info(f"{'=' * 60}")