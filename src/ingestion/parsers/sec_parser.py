# src/ingestion/parsers/sec_parser.py
"""
Parser pour les fichiers 'full-submission.txt' de la SEC EDGAR.

Améliorations :
  - Idempotence via BaseParser.parse_and_save()
  - Routing : data/interim/parsed/{company}/sec/{source}.json
  - Fix StopIteration dans _parse_sec_header (fichiers < 50 lignes)
  - Distinction Q1/Q2/Q3/Q4 pour les 10-Q (au lieu de FYxxxx systématique)
  - Détection des sections (Item, PART) pour enrichir metadata
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SECParser(BaseParser):
    """
    Parser spécialisé pour les fichiers 'full-submission.txt' de la SEC EDGAR.
    Extrait le texte lisible du dépôt principal (10-K, 10-Q, 8-K) en ignorant
    les données XBRL brutes et les pièces jointes techniques.
    """

    # Types de documents à conserver (le rapport principal, pas les EX- ni le XBRL)
    TYPES_TO_KEEP = {
        "10-K", "10-Q", "8-K", "20-F", "6-K", "SC 13G", "SC 13D",
    }

    # Patterns de sections (identiques à pdf_parser, adaptés au texte SEC)
    SECTION_PATTERNS = [
        (r"^\s*Item\s+(\d+[A-Z]?)\.\s+(.{3,80})$", "Item"),
        (r"^\s*PART\s+([IVX]+)\b(.*)$", "Part"),
    ]

    def __init__(self):
        super().__init__(document_type="sec_filing")

    # =========================================================================
    # MÉTHODE PRINCIPALE
    # =========================================================================
    def parse(self, file_path: Path) -> ParsedDocument:
        # 1. Validation
        self._validate_file(file_path, ".txt")

        # 2. Métadonnées : dossier SEC = {ticker}/{form_type}/{accession}/
        # Le ticker est 3 niveaux au-dessus : data/raw/sec/sec-edgar-filings/AAPL/10-K/0000.../
        parent_company = file_path.parent.parent.parent.name
        meta = self._extract_metadata_from_filename(file_path.name, parent_company)
        sec_meta = self._parse_sec_header(file_path)

        logger.info(
            f"📄 Parsing SEC : {file_path.name} ({sec_meta.get('type', 'Unknown')})"
        )

        # 3. Lecture complète
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()

        # 4. Extraction du texte principal
        clean_text = self._extract_main_text(raw_content)

        if not clean_text or len(clean_text) < 100:
            logger.warning(
                f"⚠️  Très peu de texte extrait pour {file_path.name}. "
                "Le document est peut-être uniquement en XBRL."
            )

        # 5. Détection des sections
        sections_found = self._detect_sections(clean_text)

        # 6. Période : on donne la priorité au header SEC (plus fiable)
        period = sec_meta.get("period") or meta["period"]

        # 7. Construction du ParsedDocument
        return ParsedDocument(
            source=file_path.stem,
            document_type=sec_meta.get("type", "sec_filing"),
            company=meta["company"],
            period=period,
            content=clean_text.strip(),
            metadata={
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path),
                "sec_accession": file_path.parent.name,
                "filing_date": sec_meta.get("filing_date", ""),
                "sections": sections_found,
            },
            chunks=[],
        )

    # =========================================================================
    # EN-TÊTE SEC (CONFORMED SUBMISSION TYPE, PERIOD, FILED AS OF DATE)
    # =========================================================================
    def _parse_sec_header(self, file_path: Path) -> dict:
        """Extrait les métadonnées SEC depuis l'en-tête du fichier."""
        # ✅ FIX : lecture sécurisée des 50 premières lignes (pas de StopIteration)
        header_lines: List[str] = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                header_lines.append(line)
        header_text = "".join(header_lines)

        doc_type = "Unknown"
        period = "Unknown"
        filing_date = ""
        report_date_raw = None

        # Type de dépôt
        m = re.search(r"CONFORMED SUBMISSION TYPE:\s*(\S+)", header_text)
        if m:
            doc_type = m.group(1).strip()

        # Date de la période rapportée (format YYYYMMDD)
        m = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header_text)
        if m:
            report_date_raw = m.group(1)

        # Date de dépôt
        m = re.search(r"FILED AS OF DATE:\s*(\d{8})", header_text)
        if m:
            d = m.group(1)
            filing_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        # ✅ FIX : déterminer period selon le type de document
        if report_date_raw:
            year = report_date_raw[:4]
            month = int(report_date_raw[4:6])

            if doc_type.upper() in ("10-Q",):
                # Pour un 10-Q, le mois indique le trimestre fiscal
                # Mois 1-3 → Q1, 4-6 → Q2, 7-9 → Q3, 10-12 → Q4 (rare)
                quarter = (month - 1) // 3 + 1
                period = f"Q{quarter}-{year}"
            elif doc_type.upper() in ("10-K", "20-F"):
                # Pour un 10-K, la date est la fin de l'exercice fiscal
                period = f"FY{year}"
            else:
                # 8-K et autres : on met l'année seule
                period = f"FY{year}"

        return {"type": doc_type, "period": period, "filing_date": filing_date}

    # =========================================================================
    # EXTRACTION DU TEXTE PRINCIPAL
    # =========================================================================
    def _extract_main_text(self, raw_content: str) -> str:
        """
        Trouve le premier document principal et nettoie son HTML.
        Les fichiers SEC contiennent plusieurs blocs <DOCUMENT>...</DOCUMENT> :
        le rapport principal (10-K, 10-Q...) et des pièces jointes (EX-xx, XML...).
        """
        documents = re.split(r"<DOCUMENT>", raw_content)

        for doc in documents[1:]:
            type_match = re.search(r"<TYPE>(.*)", doc)
            if not type_match:
                continue

            doc_type = type_match.group(1).strip().upper()
            if doc_type not in self.TYPES_TO_KEEP:
                continue

            text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
            if not text_match:
                continue

            inner_text = text_match.group(1)
            return self._clean_html(inner_text)

        # Fallback : premier <TEXT> trouvé
        first_text = re.search(r"<TEXT>(.*?)</TEXT>", raw_content, re.DOTALL)
        if first_text:
            return self._clean_html(first_text.group(1))

        return ""

    def _clean_html(self, raw: str) -> str:
        """Nettoie du HTML/XHTML SEC et normalise les espaces."""
        if "<html" in raw.lower() or "<body" in raw.lower():
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        else:
            text = raw

        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # =========================================================================
    # DÉTECTION DES SECTIONS
    # =========================================================================
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Détecte les sections Item X.Y et PART I/II/III dans le texte SEC.
        Utile pour le chunking par section en aval.
        """
        found = []
        # On limite la recherche aux 5000 premières lignes pour ne pas exploser
        for line in text.split("\n")[:5000]:
            line = line.strip()
            if len(line) < 5 or len(line) > 120:
                continue
            for pattern, section_type in self.SECTION_PATTERNS:
                m = re.match(pattern, line, re.IGNORECASE)
                if m:
                    found.append(
                        {
                            "type": section_type,
                            "label": line[:100],
                        }
                    )
                    break
        return found

    def _compute_output_key(self, source_key: str) -> str:
        """
        source_key = 'raw-documents/sec/sec-edgar-filings/AAPL/10-K/0000.../full-submission.txt'
        → 'parsed-documents/aapl/sec/10-k/full-submission.json'
        """
        parts = source_key.split("/")
        filename = parts[-1]
        # company = 3 niveaux au-dessus (indice -4)
        parent_company = parts[-4] if len(parts) >= 4 else ""
        meta = self._extract_metadata_from_filename(filename, parent_company)
        company_slug = (meta["company"] or "unknown").lower()

        # doc_type = 2 niveaux au-dessus (indice -3)
        raw_type = parts[-3].lower() if len(parts) >= 3 else "sec"
        type_slug = re.sub(r"[^\w]", "-", raw_type)

        stem = Path(filename).stem
        return f"parsed-documents/{company_slug}/sec/{type_slug}/{stem}.json"


# ==============================================================================
# EXÉCUTION DIRECTE (idempotente + routing)
# ==============================================================================
if __name__ == "__main__":
    from src.core.storage import storage
    from src.ingestion.parsers.manifest import load_manifest, save_manifest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = SECParser()
    manifest = load_manifest()

    source_keys = [
        k for k in storage.list("raw-documents/sec/")
        if k.lower().endswith("full-submission.txt")
    ]
    logger.info(f"🔍 {len(source_keys)} filings SEC à parser")

    stats = {"parsed": 0, "skipped": 0, "failed": 0}
    for src in source_keys:
        try:
            doc = parser.parse_and_save_to_storage(src, manifest)
            if doc is None:
                stats["skipped"] += 1
                continue
            logger.info(
                f"✅ [{doc.company}] {src} | Type={doc.document_type} | "
                f"Period={doc.period} | Chars={len(doc.content):,}"
            )
            stats["parsed"] += 1
        except Exception as e:
            logger.error(f"❌ Échec {src} : {e}")
            stats["failed"] += 1

    save_manifest(manifest)
    logger.info(
        f"\n🎉 SEC : parsed={stats['parsed']} | "
        f"skipped={stats['skipped']} | failed={stats['failed']}"
    )