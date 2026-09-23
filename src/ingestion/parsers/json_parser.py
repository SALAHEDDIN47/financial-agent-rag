# src/ingestion/parsers/json_parser.py
"""
Parser pour fichiers JSON : Transcripts, News, FinQA, et JSON génériques.

Spécificité : un fichier source peut produire PLUSIEURS ParsedDocument
(ex : AAPL_news.json → 10 articles). On utilise donc `parse_file_and_save()`
au lieu de `parse_and_save()`.

Idempotence :
  - Le manifest stocke la liste complète des outputs pour chaque source.
  - Si la source change, les anciens outputs sont supprimés avant regénération
    (pour éviter les orphelins si le nombre d'items diminue).

Organisation de sortie :
  data/interim/parsed/{company}/{doc_type_slug}/{source}.json
  ex : data/interim/parsed/aapl/finqa/AAPL_2015_page_68.pdf-3.json
       data/interim/parsed/msft/news/MSFT_news_article_0.json
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from src.ingestion.parsers.base_parser import (
    BaseParser,
    ParsedDocument,
    compute_file_hash,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# MAPPING document_type → nom de dossier
# ==============================================================================
DOC_TYPE_TO_FOLDER = {
    "earnings_transcript": "transcripts",
    "news": "news",
    "finqa_qa_pair": "finqa",
    "generic_json": "generic",
    "empty_json": "empty",
}


def _slugify_document_type(doc_type: str) -> str:
    """Convertit un document_type en nom de dossier propre."""
    return DOC_TYPE_TO_FOLDER.get(doc_type, doc_type.replace("_", "-"))


class JSONParser(BaseParser):
    """Parser spécialisé pour les fichiers JSON (Transcripts, News, FinQA)."""

    def __init__(self, finqa_max_items: int = 5000):
        """
        Paramètres
        ----------
        finqa_max_items : int
            Nombre maximum d'entrées FinQA à traiter par fichier.
            FinQA train.json contient ~6000 items, on limite par défaut à 5000
            pour éviter de saturer la sortie. Mettre à -1 pour tout traiter.
        """
        super().__init__(document_type="json_data")
        self.finqa_max_items = finqa_max_items

    # =========================================================================
    # CONTRAT ABSTRAIT (retourne le 1er doc uniquement)
    # =========================================================================
    def parse(self, file_path: Path) -> ParsedDocument:
        """Retourne le premier document parsé (respect du contrat BaseParser)."""
        docs = self.parse_file(file_path)
        if docs:
            return docs[0]

        return ParsedDocument(
            source=file_path.stem,
            document_type="empty_json",
            company="UNKNOWN",
            period="UNKNOWN",
            content="",
            metadata={"file_path": str(file_path)},
        )

    # =========================================================================
    # PARSING PRINCIPAL (retourne la LISTE complète)
    # =========================================================================
    def parse_file(self, file_path: Path) -> List[ParsedDocument]:
        """
        Parse un fichier JSON et retourne une liste de ParsedDocument.
        Le routing se fait sur le nom de fichier ET le chemin complet
        (pour détecter FinQA même quand le fichier s'appelle juste dev.json).
        """
        self._validate_file(file_path, ".json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de lecture JSON dans {file_path.name}: {e}")
            return []

        filename = file_path.name.lower()
        path_str = str(file_path).lower().replace("\\", "/")

        if "transcript" in filename:
            documents = self._parse_transcript(data, file_path)
        elif "news" in filename:
            documents = self._parse_news(data, file_path)
        elif (
            "finqa" in path_str
            or filename in ("dev.json", "train.json", "test.json")
        ):
            documents = self._parse_finqa(data, file_path)
        else:
            documents = self._parse_generic(data, file_path)

        logger.debug(f"✅ {file_path.name} → {len(documents)} document(s)")
        return documents

    # =========================================================================
    # WRAPPER IDEMPOTENT (multi-documents)
    # =========================================================================
    def parse_file_and_save(
        self,
        source: Path,
        output_dir: Path,
        manifest: dict,
        force: bool = False,
    ) -> List[ParsedDocument]:
        """
        Version idempotente de parse_file() pour les sources multi-documents.

        Comportement :
          1. Si source inchangée (même hash) ET tous les outputs existent → retourne [].
          2. Sinon : parse, SUPPRIME les anciens outputs, regénère tout, met à jour le manifest.

        Les fichiers sont rangés par :
          output_dir / {company}/{doc_type_slug}/{source}.json
        """
        key = str(source)
        prev = manifest.get(key)

        # 1. Vérification idempotente
        if not force and prev and output_dir.exists():
            try:
                current_hash = compute_file_hash(source)
            except (FileNotFoundError, PermissionError):
                current_hash = None

            if current_hash and current_hash == prev.get("source_hash"):
                outputs_ok = all(
                    Path(p).exists() for p in prev.get("outputs", [])
                )
                if outputs_ok:
                    logger.debug(f"⏭️  Ignoré (déjà parsé) : {source.name}")
                    return []

        # 2. Nettoyage des anciens outputs
        if prev and "outputs" in prev:
            for old_path_str in prev["outputs"]:
                old_path = Path(old_path_str)
                if old_path.exists():
                    try:
                        old_path.unlink()
                    except OSError as e:
                        logger.warning(f"Impossible de supprimer {old_path}: {e}")

        # 3. Parsing
        docs = self.parse_file(source)

        source_hash = compute_file_hash(source) if source.exists() else None
        parsed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        if not docs:
            manifest[key] = {
                "source_hash": source_hash,
                "parsed_at": parsed_at,
                "outputs": [],
                "count": 0,
                "document_types": [],
            }
            return []

        # 4. Sauvegarde ROUTÉE par company/doc_type
        outputs = []
        doc_types = set()

        for doc in docs:
            doc.metadata["source_hash"] = source_hash
            doc.metadata["parsed_at"] = parsed_at
            doc_types.add(doc.document_type)

            # ✅ Routing : {company}/{doc_type_slug}/
            company_slug = (doc.company or "unknown").lower()
            type_slug = _slugify_document_type(doc.document_type)
            target_dir = output_dir / company_slug / type_slug
            target_dir.mkdir(parents=True, exist_ok=True)

            safe_source = re.sub(r"[^\w\-_.]", "_", doc.source)
            out_file = target_dir / f"{safe_source}.json"

            self.save_document(doc, out_file)
            outputs.append(str(out_file))

        # 5. Mise à jour du manifest
        manifest[key] = {
            "source_hash": source_hash,
            "parsed_at": parsed_at,
            "outputs": outputs,
            "count": len(docs),
            "document_types": sorted(doc_types),
        }

        return docs

    # =========================================================================
    # PARSERS SPÉCIALISÉS
    # =========================================================================
    def _parse_transcript(
        self, data: Dict[str, Any], file_path: Path
    ) -> List[ParsedDocument]:
        """Parse les transcripts d'earnings calls."""
        docs = []
        meta = self._extract_metadata_from_filename(file_path.name)
        company = str(data.get("ticker") or meta["company"]).upper()

        quarter = data.get("quarter", "UNKNOWN")
        year = data.get("year", "UNKNOWN")
        period = f"Q{quarter}-{year}" if quarter != "UNKNOWN" else "UNKNOWN"

        content = data.get("transcript", "")
        if not content and isinstance(data, dict):
            candidates = [
                str(v)
                for v in data.values()
                if isinstance(v, str) and len(v) > 1000
            ]
            content = max(candidates, key=len, default="")

        if content:
            docs.append(
                ParsedDocument(
                    source=file_path.stem,
                    document_type="earnings_transcript",
                    company=company,
                    period=period,
                    content=content.strip(),
                    metadata={
                        "date_published": data.get("datePublished", ""),
                        "file_path": str(file_path),
                    },
                    chunks=[],
                )
            )
        return docs

    def _parse_news(
        self, data: List[Dict[str, Any]], file_path: Path
    ) -> List[ParsedDocument]:
        """Parse les flux d'actualités (liste d'articles)."""
        docs = []
        company = self._extract_metadata_from_filename(file_path.name)["company"]

        for idx, article in enumerate(data):
            content_data = article.get("content", {})
            title = content_data.get("title", "").strip()
            summary = content_data.get("summary", "").strip()

            # Skip les placeholders Yahoo
            if not title and not summary:
                continue

            pub_date = content_data.get("pubDate", "")
            provider = content_data.get("provider", {}).get("displayName", "Unknown")
            url = content_data.get("canonicalUrl", {}).get("url", "")

            text_content = (
                f"TITLE: {title}\n"
                f"DATE: {pub_date}\n"
                f"SOURCE: {provider}\n"
                f"SUMMARY: {summary}\n"
                f"URL: {url}"
            )

            docs.append(
                ParsedDocument(
                    source=f"{file_path.stem}_article_{idx}",
                    document_type="news",
                    company=company,
                    period="LATEST",
                    content=text_content.strip(),
                    metadata={
                        "title": title,
                        "url": url,
                        "pub_date": pub_date,
                        "provider": provider,
                        "file_path": str(file_path),
                    },
                    chunks=[],
                )
            )
        return docs

    def _parse_finqa(
        self, data: List[Dict[str, Any]], file_path: Path
    ) -> List[ParsedDocument]:
        """Parse le dataset FinQA (paires QA + contexte tabulaire)."""
        docs = []

        if self.finqa_max_items < 0:
            max_items = len(data)
        else:
            max_items = min(len(data), self.finqa_max_items)

        logger.info(f"FinQA : traitement de {max_items}/{len(data)} entrées")

        for idx, item in enumerate(data[:max_items]):
            qa = item.get("qa", {})
            table = item.get("table", [])
            pre_text = " ".join(item.get("pre_text", []))
            post_text = " ".join(item.get("post_text", []))

            context_text = f"CONTEXT TEXT: {pre_text}\n\n"
            if table:
                table_str = "\n".join(
                    " | ".join(str(c) for c in row) for row in table
                )
                context_text += f"TABLE DATA:\n{table_str}\n\n"
            context_text += f"POST TEXT: {post_text}"

            final_content = (
                f"QUESTION: {qa.get('question', '')}\n"
                f"ANSWER: {qa.get('answer', '')}\n"
                f"REASONING: {qa.get('explanation', '')}\n"
                f"FORMULA: {qa.get('program', '')}\n\n"
                f"--- SUPPORTING CONTEXT ---\n{context_text}"
            )

            doc_id = str(item.get("id", f"finqa_{idx}")).replace("/", "_")

            # ✅ FIX : extraction du ticker depuis l'ID FinQA
            company = self._extract_company_from_finqa_id(doc_id)

            docs.append(
                ParsedDocument(
                    source=doc_id,
                    document_type="finqa_qa_pair",
                    company=company,
                    period="UNKNOWN",
                    content=final_content.strip(),
                    metadata={
                        "original_id": item.get("id"),
                        "has_table": len(table) > 0,
                        "file_path": str(file_path),
                    },
                    chunks=[],
                )
            )
        return docs

    def _parse_generic(self, data: Any, file_path: Path) -> List[ParsedDocument]:
        """Fallback pour tout autre JSON non identifié."""
        meta = self._extract_metadata_from_filename(file_path.name)
        return [
            ParsedDocument(
                source=file_path.stem,
                document_type="generic_json",
                company=meta["company"],
                period=meta["period"],
                content=json.dumps(data, indent=2, ensure_ascii=False),
                metadata={"file_path": str(file_path)},
                chunks=[],
            )
        ]

    # =========================================================================
    # HELPERS
    # =========================================================================
    @staticmethod
    def _extract_company_from_finqa_id(item_id: str) -> str:
        """
        Extrait le ticker depuis un ID FinQA.
        Ex : 'AAPL_2015_page_68.pdf-3' → 'AAPL'
             'AAL_2014_page_89.pdf-1'  → 'AAL'
             'AES_2001_page_85.pdf-4'  → 'AES'
        """
        if not item_id:
            return "UNKNOWN"
        match = re.match(r"^([A-Z]{1,5})_", item_id.upper())
        if match:
            return match.group(1)
        return "UNKNOWN"


# ==============================================================================
# EXÉCUTION DIRECTE (avec idempotence via manifest)
# ==============================================================================
if __name__ == "__main__":
    import sys
    import time

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.ingestion.parsers.manifest import load_manifest, save_manifest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = JSONParser()

    raw_dir = Path("data/raw")
    output_dir = Path("data/interim/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_dir.rglob("*.json"))
    logger.info(f"🔍 {len(json_files)} fichiers JSON trouvés dans {raw_dir}")

    manifest = load_manifest()
    stats = {"parsed_files": 0, "skipped_files": 0, "parsed_docs": 0, "failed": 0}

    # --- Checkpoint state ---
    SAVE_EVERY = 100
    SAVE_INTERVAL_S = 60
    ckpt = {"count": 0, "last_save": time.time()}

    def _checkpoint():
        now = time.time()
        if (ckpt["count"] >= SAVE_EVERY or 
            (now - ckpt["last_save"]) >= SAVE_INTERVAL_S):
            save_manifest(manifest)
            ckpt["count"] = 0
            ckpt["last_save"] = now
            logger.debug(f"💾 Checkpoint : {len(manifest)} entrées")

    interrupted = False
    try:
        for json_file in json_files:
            try:
                docs = parser.parse_file_and_save(
                    source=json_file,
                    output_dir=output_dir,
                    manifest=manifest,
                )
            except Exception as e:
                logger.error(f"❌ Échec : {json_file.name} → {e}")
                stats["failed"] += 1
                continue

            if docs:
                stats["parsed_files"] += 1
                stats["parsed_docs"] += len(docs)
                doc_types = sorted(set(d.document_type for d in docs))
                companies = sorted(set(d.company for d in docs))
                logger.info(
                    f"✅ {json_file.name} → {len(docs)} doc(s) "
                    f"[{', '.join(doc_types)}] companies={companies}"
                )
                ckpt["count"] += 1
                _checkpoint()
            else:
                stats["skipped_files"] += 1
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("\n⏸️  Ctrl+C détecté — sauvegarde du manifest...")
    finally:
        save_manifest(manifest)
        logger.info(f"💾 Manifest sauvegardé : {len(manifest)} entrées")

    if interrupted:
        logger.info("💡 Pour reprendre : uv run ./src/ingestion/parsers/json_parser.py")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎉 Parsing JSON terminé !")
    logger.info(f"   ✅ Fichiers parsés   : {stats['parsed_files']}")
    logger.info(f"   ⏭️  Fichiers ignorés : {stats['skipped_files']}")
    logger.info(f"   📊 Total docs créés  : {stats['parsed_docs']}")
    logger.info(f"   ❌ Échecs            : {stats['failed']}")
    logger.info(f"   📁 Sortie            : {output_dir.absolute()}")
    logger.info(f"{'=' * 60}")