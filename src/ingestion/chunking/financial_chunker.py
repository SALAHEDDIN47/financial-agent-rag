# src/ingestion/chunking/financial_chunker.py
"""
Chunker financier avec :
  - Idempotence via manifest (cohérent avec les parsers)
  - Chunking par TOKENS (adapté à Qwen 2.5)
  - Protection des tableaux Markdown (jamais coupés en plein milieu)
  - Propagation du numéro de page (depuis les marqueurs `--- PAGE N ---`)
  - Routing par `{company}/{doc_type}/`
  - Métadonnées enrichies pour le filtrage RAG
  - ✅ FIX 1 : chunk_id UNIQUE (hash du chemin inclus)
  - ✅ FIX 2 : source exposé au niveau RACINE (pour indexers Milvus/ES)
  - ✅ Checkpoints périodiques (tous les 100 fichiers OU 60s)
  - ✅ Ctrl+C géré proprement

Usage :
    uv run ./src/ingestion/chunking/financial_chunker.py
    uv run ./src/ingestion/chunking/financial_chunker.py --force  # re-chunke tout
"""
import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Compteur de tokens (tiktoken recommandé, fallback heuristique) ----------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    TOKENIZER_NAME = "tiktoken/cl100k_base"
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)
    TOKENIZER_NAME = "heuristic (len/4)"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# MAPPING document_type → dossier
# ==============================================================================
DOC_TYPE_TO_FOLDER = {
    "10-K": "10k",
    "10-Q": "10q",
    "8-K": "8k",
    "presentation_slides": "presentations",
    "proxy_statement": "proxy",
    "shareholder_letter": "shareholder-letters",
    "financial_report": "reports",
    "earnings_transcript": "transcripts",
    "news": "news",
    "finqa_qa_pair": "finqa",
    "generic_json": "generic",
}


def _slugify_document_type(doc_type: str) -> str:
    return DOC_TYPE_TO_FOLDER.get(doc_type, re.sub(r"[^\w]", "-", doc_type.lower()))


def _short_hash(text: str, length: int = 8) -> str:
    """Hash court d'une chaîne (pour rendre les chunk_id uniques)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


# ==============================================================================
# SEGMENTATION EN UNITÉS ATOMIQUES
# ==============================================================================
PAGE_MARKER_RE = re.compile(r"^---\s*PAGE\s+(\d+)\s*---\s*$", re.IGNORECASE)


def _segment_into_units(text: str) -> List[Dict[str, Any]]:
    """
    Segmente le contenu d'un document en unités atomiques :
      - paragraph : bloc de lignes de texte
      - table     : tableau Markdown complet (ne DOIT pas être coupé)
    """
    units: List[Dict[str, Any]] = []
    current_page: Optional[int] = None
    buffer: List[str] = []

    def flush_buffer():
        if buffer:
            units.append(
                {
                    "type": "paragraph",
                    "text": "\n".join(buffer).strip(),
                    "page": current_page,
                }
            )
            buffer.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. Marqueur de page
        m = PAGE_MARKER_RE.match(line.strip())
        if m:
            flush_buffer()
            current_page = int(m.group(1))
            i += 1
            continue

        # 2. Début d'un tableau Markdown
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
        ):
            flush_buffer()
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            units.append(
                {
                    "type": "table",
                    "text": "\n".join(table_lines),
                    "page": current_page,
                }
            )
            continue

        # 3. Ligne vide → fin de paragraphe
        if not stripped:
            flush_buffer()
            i += 1
            continue

        buffer.append(line)
        i += 1

    flush_buffer()
    return units


# ==============================================================================
# REGROUPEMENT DES UNITÉS EN CHUNKS
# ==============================================================================
def _split_large_unit(
    unit: Dict[str, Any],
    max_tokens: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    """Coupe une unité trop grosse (ex : un tableau de 5000 tokens) en morceaux."""
    text = unit["text"]
    lines = text.split("\n")
    sub_chunks = []
    current_lines = []
    current_tokens = 0

    header = lines[0] if unit["type"] == "table" and lines else None

    for line in lines:
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > max_tokens and current_lines:
            sub_chunks.append(
                {
                    "type": unit["type"],
                    "text": "\n".join(current_lines),
                    "page": unit["page"],
                }
            )
            current_lines = [header] if header and header not in current_lines else []
            current_tokens = count_tokens("\n".join(current_lines)) if current_lines else 0
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        sub_chunks.append(
            {
                "type": unit["type"],
                "text": "\n".join(current_lines),
                "page": unit["page"],
            }
        )
    return sub_chunks


def _group_units_into_chunks(
    units: List[Dict[str, Any]],
    max_tokens: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    """Regroupe les unités en chunks respectant max_tokens, avec overlap."""
    chunks: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_tokens = 0

    def flush_current():
        if not current_units:
            return
        text = "\n\n".join(u["text"] for u in current_units)
        page = next((u["page"] for u in current_units if u["page"] is not None), None)
        chunks.append({"text": text, "page": page, "unit_count": len(current_units)})

    for unit in units:
        unit_tokens = count_tokens(unit["text"])

        if unit_tokens > max_tokens:
            flush_current()
            current_units = []
            current_tokens = 0
            sub = _split_large_unit(unit, max_tokens, overlap_tokens)
            for s in sub:
                chunks.append({"text": s["text"], "page": s["page"], "unit_count": 1})
            continue

        if current_tokens + unit_tokens > max_tokens and current_units:
            flush_current()
            overlap_units: List[Dict[str, Any]] = []
            overlap_acc = 0
            for u in reversed(current_units):
                u_tok = count_tokens(u["text"])
                if overlap_acc + u_tok > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_acc += u_tok
            current_units = overlap_units
            current_tokens = overlap_acc

        current_units.append(unit)
        current_tokens += unit_tokens

    flush_current()
    return chunks


# ==============================================================================
# CHUNKER PRINCIPAL
# ==============================================================================
class FinancialChunker:
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 80,
        min_chunk_tokens: int = 20,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Découpe un document parsé (dict) en chunks enrichis."""
        content = doc.get("content", "")
        if not content or len(content) < 50:
            return []

        # 1. Segmentation
        units = _segment_into_units(content)
        if not units:
            return []

        # 2. Regroupement
        chunks = _group_units_into_chunks(units, self.max_tokens, self.overlap_tokens)

        # 3. Métadonnées
        doc_metadata = doc.get("metadata", {}) or {}
        source = doc.get("source", "unknown")
        company = doc.get("company", "UNKNOWN")
        period = doc.get("period", "UNKNOWN")
        document_type = doc.get("document_type", "UNKNOWN")
        file_path = doc_metadata.get("file_path", "")

        # ✅ FIX 1 : hash du chemin pour rendre les chunk_id uniques
        # Deux fichiers "full-submission.json" (AAPL 10-K et TSLA 10-K) produiront
        # des chunk_id différents : full-submission_A3F8E2C1_0000 vs full-submission_9B7D41E5_0000
        path_hash = _short_hash(file_path) if file_path else _short_hash(source)

        enriched: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            text = ch["text"].strip()
            if count_tokens(text) < self.min_chunk_tokens:
                continue

            is_table = text.lstrip().startswith("|")

            enriched.append(
                {
                    # ✅ FIX 1 : chunk_id globalement unique
                    "chunk_id": f"{source}_{path_hash}_{i:04d}",
                    "text": text,
                    # Métadonnées de haut niveau
                    "company": company,
                    "period": period,
                    "document_type": document_type,
                    # ✅ FIX 2 : source au niveau RACINE (pour indexers)
                    "source": source,
                    "page": ch["page"],
                    "metadata": {
                        "source": source,          # conservé aussi dans metadata
                        "file_path": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "token_count": count_tokens(text),
                        "is_table": is_table,
                    },
                }
            )
        return enriched


# ==============================================================================
# ORCHESTRATION + IDEMPOTENCE
# ==============================================================================
MANIFEST_PATH = Path("data/processed/.chunking_manifest.json")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("⚠️  Manifest corrompu, redémarrage à vide")
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    """Sauvegarde atomique."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(MANIFEST_PATH)


def _compute_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_documents(
    input_dir: Path,
    output_dir: Path,
    chunker: FinancialChunker,
    force: bool = False,
    save_every: int = 100,
    save_interval_s: int = 60,
) -> Dict[str, int]:
    """
    Parcourt tous les JSON parsés, les chunke, sauvegarde en JSONL (idempotent).
    ✅ Checkpoints périodiques (N fichiers OU X secondes).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    json_files = sorted(input_dir.rglob("*.json"))
    logger.info(f"🔍 {len(json_files)} fichiers JSON trouvés dans {input_dir}")

    stats = {
        "parsed": 0,
        "skipped": 0,
        "failed": 0,
        "empty": 0,
        "total_chunks": 0,
    }

    # --- Checkpoint state ---
    ckpt = {"count": 0, "last_save": time.time()}

    def _checkpoint():
        now = time.time()
        if (ckpt["count"] >= save_every or 
            (now - ckpt["last_save"]) >= save_interval_s):
            _save_manifest(manifest)
            ckpt["count"] = 0
            ckpt["last_save"] = now
            logger.debug(f"💾 Checkpoint chunking : {len(manifest)} entrées")

    interrupted = False
    try:
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                company_slug = (doc.get("company") or "unknown").lower()
                type_slug = _slugify_document_type(doc.get("document_type", "unknown"))
                target_dir = output_dir / company_slug / type_slug
                target_dir.mkdir(parents=True, exist_ok=True)

                output_file = target_dir / f"{json_file.stem}_chunks.jsonl"

                key = str(json_file)
                prev = manifest.get(key)
                if not force and prev and output_file.exists():
                    try:
                        current_hash = _compute_hash(json_file)
                    except (FileNotFoundError, PermissionError):
                        current_hash = None
                    if current_hash and current_hash == prev.get("source_hash"):
                        stats["skipped"] += 1
                        continue

                chunks = chunker.chunk_document(doc)
                if not chunks:
                    stats["empty"] += 1
                    manifest[key] = {
                        "source_hash": _compute_hash(json_file),
                        "chunked_at": datetime.now(timezone.utc)
                            .isoformat().replace("+00:00", "Z"),
                        "count": 0,
                        "output": str(output_file),
                    }
                    ckpt["count"] += 1
                    _checkpoint()
                    continue

                with open(output_file, "w", encoding="utf-8") as f_out:
                    for chunk in chunks:
                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                manifest[key] = {
                    "source_hash": _compute_hash(json_file),
                    "chunked_at": datetime.now(timezone.utc)
                        .isoformat().replace("+00:00", "Z"),
                    "count": len(chunks),
                    "output": str(output_file),
                }
                stats["parsed"] += 1
                stats["total_chunks"] += len(chunks)
                ckpt["count"] += 1
                _checkpoint()

            except Exception as e:
                logger.error(f"❌ Échec : {json_file.name} → {e}")
                stats["failed"] += 1
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("\n⏸️  Ctrl+C détecté — sauvegarde du manifest...")
    finally:
        _save_manifest(manifest)
        logger.info(f"💾 Manifest sauvegardé : {len(manifest)} entrées")

    if interrupted:
        logger.info("💡 Pour reprendre : uv run ./src/ingestion/chunking/financial_chunker.py")

    return stats


# ==============================================================================
# EXÉCUTION DIRECTE
# ==============================================================================
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Force le rechunking")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/parsed"),
        help="Dossier d'entrée (JSON parsés)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks"),
        help="Dossier de sortie (JSONL chunks)",
    )
    args = ap.parse_args()

    chunker = FinancialChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
    )

    logger.info(
        f"⚙️  Chunker : max_tokens={args.max_tokens}, overlap={args.overlap}, "
        f"tokenizer={TOKENIZER_NAME}"
    )

    t0 = time.time()
    stats = chunk_documents(
        input_dir=args.input,
        output_dir=args.output,
        chunker=chunker,
        force=args.force,
    )
    elapsed = time.time() - t0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Chunking terminé !")
    logger.info(f"   ✅ Fichiers chunkés   : {stats['parsed']}")
    logger.info(f"   ⏭️  Fichiers ignorés  : {stats['skipped']}")
    logger.info(f"   📭 Docs vides         : {stats['empty']}")
    logger.info(f"   ❌ Échecs             : {stats['failed']}")
    logger.info(f"   📊 Total chunks créés : {stats['total_chunks']}")
    logger.info(f"   ⏱️  Temps total       : {elapsed:.1f}s")
    logger.info(f"   📁 Sortie             : {args.output.absolute()}")
    logger.info(f"{'=' * 60}")