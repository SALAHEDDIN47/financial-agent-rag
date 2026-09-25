# src/ingestion/parsers/pdf_parser.py
"""
Parser PDF (pdfplumber).

Améliorations :
  - Idempotence via BaseParser.parse_and_save()
  - Routing : data/interim/parsed/{company}/{doc_type_slug}/{source}.json
  - Fix table_idx (affichage 1-based, cohérent avec la liste tables_data)
  - Détection des sections (Item X, PART I, Risk Factors) → metadata["sections"]
  - Option preserve_tables, extract_layout
  - Fallback warning si PDF probablement scanné
  - ✅ FIX : normalisation des tirets/underscores pour inférer le bon type
  - ✅ FIX : silence des warnings pdfminer (couleurs CMYK non standard)
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ✅ FIX : silence les warnings bruyants de pdfminer (couleurs CMYK/RGB non standard)
# Ces warnings n'empêchent pas le parsing, ils polluent juste les logs.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


# Mapping document_type → slug de dossier
DOC_TYPE_TO_FOLDER = {
    "10-K": "10k",
    "10-Q": "10q",
    "presentation_slides": "presentations",
    "proxy_statement": "proxy",
    "shareholder_letter": "shareholder-letters",
    "financial_report": "reports",
}


def _slugify_document_type(doc_type: str) -> str:
    return DOC_TYPE_TO_FOLDER.get(doc_type, re.sub(r"[^\w]", "-", doc_type.lower()))


class PDFParser(BaseParser):
    def __init__(self, preserve_tables: bool = True, extract_layout: bool = False):
        """
        Paramètres
        ----------
        preserve_tables : bool
            Si True, extrait les tableaux et les ajoute en Markdown en fin de contenu.
        extract_layout : bool
            Si True, utilise `extract_text(layout=True)` pour préserver les colonnes.
            ⚠️ Plus lent et insère beaucoup d'espaces ; à activer seulement si nécessaire.
        """
        super().__init__(document_type="pdf_document")
        self.preserve_tables = preserve_tables
        self.extract_layout = extract_layout

    # =========================================================================
    # MÉTHODE PRINCIPALE
    # =========================================================================
    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path, ".pdf")

        # Métadonnées depuis le chemin (dossier parent = entreprise)
        meta = self._extract_metadata_from_filename(
            file_path.name, file_path.parent.name
        )

        logger.info(f"📄 Parsing PDF : {file_path.name} (dossier: {file_path.parent.name})")

        full_text = ""
        tables_data: List[Dict[str, Any]] = []
        sections_found: List[Dict[str, Any]] = []
        total_pages = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page in pdf.pages:
                # Extraction du texte
                if self.extract_layout:
                    text = page.extract_text(layout=True) or ""
                else:
                    text = page.extract_text() or ""

                text = re.sub(r"\n{3,}", "\n\n", text).strip()

                if text:
                    full_text += f"\n\n--- PAGE {page.page_number} ---\n{text}"

                    # Détection des sections (Item X, PART I, Risk Factors, ...)
                    page_sections = self._detect_sections(text, page.page_number)
                    if page_sections:
                        sections_found.extend(page_sections)

                # Extraction des tableaux
                if self.preserve_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            table_md = self._table_to_markdown(table)
                            tables_data.append(
                                {
                                    "page": page.page_number,
                                    # ✅ FIX : numérotation 1-based cohérente avec l'affichage
                                    "table_idx": len(tables_data) + 1,
                                    "content": table_md,
                                }
                            )

        # Ajout des tableaux en Markdown en fin de contenu
        if tables_data:
            full_text += "\n\n" + "=" * 50 + "\n"
            full_text += "DONNÉES TABULAIRES EXTRAITES (Format Markdown)\n"
            full_text += "=" * 50 + "\n"
            for tbl in tables_data:
                full_text += f"\n[Source: Page {tbl['page']}, Tableau {tbl['table_idx']}]\n"
                full_text += tbl["content"] + "\n"

        # Détection PDF probablement scanné (très peu de texte extrait)
        if total_pages > 0 and len(full_text.strip()) < 200:
            logger.warning(
                f"⚠️  {file_path.name} : très peu de texte extrait "
                f"({len(full_text)} car. pour {total_pages} pages) — PDF probablement scanné."
            )

        doc_type = self._infer_document_type(file_path.name)

        return ParsedDocument(
            source=file_path.stem,
            document_type=doc_type,
            company=meta["company"],
            period=meta["period"],
            content=full_text.strip(),
            metadata={
                "total_pages": total_pages,
                "tables_found": len(tables_data),
                "sections": sections_found,
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path),
            },
            chunks=[],
        )

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        """Convertit un tableau pdfplumber en Markdown, en normalisant la largeur."""
        if not table:
            return ""

        cleaned = [
            [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
            for row in table
        ]
        max_cols = max(len(row) for row in cleaned)
        for row in cleaned:
            row.extend([""] * (max_cols - len(row)))

        header = cleaned[0]
        separator = ["---"] * max_cols
        rows = cleaned[1:]

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)

    # Patterns de sections financières typiques (10-K, 10-Q, rapports annuels)
    SECTION_PATTERNS = [
        # "Item 7. MD&A" / "Item 1A. Risk Factors" / "Item 8."
        (r"^\s*Item\s+(\d+[A-Z]?)\.\s+(.{3,80})$", "Item"),
        # "PART I" / "PART II" / "PART IV"
        (r"^\s*PART\s+([IVX]+)\b(.*)$", "Part"),
        # Titres récurrents (anglais)
        (r"^\s*(Risk Factors|Management'?s Discussion|Financial Statements|"
         r"Quantitative and Qualitative Disclosures|Controls and Procedures)"
         r"\b.{0,60}$", "Section"),
    ]

    def _detect_sections(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Détecte les titres de sections dans le texte d'une page.
        Retourne une liste de {type, label, page}.
        """
        found = []
        for line in page_text.split("\n"):
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
                            "page": page_num,
                        }
                    )
                    break  # une seule détection par ligne
        return found

    def _infer_document_type(self, filename: str) -> str:
        name = filename.lower().replace("-", " ").replace("_", " ")

        if ("10 k" in name or "10k" in name
                or "annual report" in name or "annualreport" in name):
            return "10-K"
        elif ("10 q" in name or "10q" in name
              or "earnings release" in name or "quarterly" in name):
            return "10-Q"
        elif "slides" in name or "presentation" in name or "deck" in name:
            return "presentation_slides"
        elif "proxy" in name:
            return "proxy_statement"
        elif "shareholder letter" in name or "shareholderletter" in name:
            return "shareholder_letter"
        elif "update" in name:
            return "10-Q"
        return "financial_report"

    def _compute_output_key(self, source_key: str) -> str:
        parts = source_key.split("/")
        filename = parts[-1]
        parent_folder = parts[-2] if len(parts) >= 2 else ""
        meta = self._extract_metadata_from_filename(filename, parent_folder)
        doc_type = self._infer_document_type(filename)
        company_slug = (meta["company"] or "unknown").lower()
        type_slug = _slugify_document_type(doc_type)
        stem = Path(filename).stem
        return f"parsed-documents/{company_slug}/{type_slug}/{stem}.json"


# ==============================================================================
# EXÉCUTION DIRECTE (idempotente via manifest + routing par company/doc_type)
# ==============================================================================
if __name__ == "__main__":
    from src.core.storage import storage
    from src.ingestion.parsers.manifest import load_manifest, save_manifest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = PDFParser(preserve_tables=True, extract_layout=False)
    manifest = load_manifest()

    # Tous les PDFs sous raw-documents/presentations/
    source_keys = [
        k for k in storage.list("raw-documents/presentations/")
        if k.lower().endswith(".pdf")
    ]
    logger.info(f"🔍 {len(source_keys)} PDFs à parser")

    stats = {"parsed": 0, "skipped": 0, "failed": 0}
    for src in source_keys:
        try:
            doc = parser.parse_and_save_to_storage(src, manifest)
            if doc is None:
                stats["skipped"] += 1
                continue
            logger.info(
                f"✅ [{doc.company}] {src} | "
                f"Type={doc.document_type} | "
                f"Pages={doc.metadata.get('total_pages')} | "
                f"Tables={doc.metadata.get('tables_found')}"
            )
            stats["parsed"] += 1
        except Exception as e:
            logger.error(f"❌ Échec {src} : {e}")
            stats["failed"] += 1

    save_manifest(manifest)
    logger.info(
        f"\n🎉 PDF : parsed={stats['parsed']} | "
        f"skipped={stats['skipped']} | failed={stats['failed']}"
    )