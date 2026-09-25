# src/ingestion/parsers/docx_parser.py
"""
Parser pour les documents Word (.docx) : rapports annuels Microsoft,
lettres aux actionnaires, 10-K Word.

Améliorations par rapport à la version précédente :
  - Détection des titres (Heading 1/2/3) → balises Markdown (#, ##, ###)
  - Gestion correcte des cellules fusionnées (déduplication par identité XML)
  - Intégration de l'idempotence via BaseParser.parse_and_save()
  - Ajout de source_hash et parsed_at dans metadata
"""
import logging
from pathlib import Path

from docx import Document
from docx.table import Table

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class DOCXParser(BaseParser):
    """
    Parser spécialisé pour les rapports annuels et lettres aux actionnaires (.docx).
    """

    # Mapping des styles Word vers des niveaux Markdown
    HEADING_STYLES = {
        "title": "#",
        "heading 1": "##",
        "heading 2": "###",
        "heading 3": "####",
        "heading 4": "#####",
        "subtitle": "###",
    }

    def __init__(self):
        super().__init__(document_type="docx_report")

    # -------------------------------------------------------------------------
    # MÉTHODE PRINCIPALE
    # -------------------------------------------------------------------------
    def parse(self, file_path: Path) -> ParsedDocument:
        # 1. Validation
        self._validate_file(file_path, ".docx")

        # 2. Métadonnées (company + period)
        meta = self._extract_metadata_from_filename(
            file_path.name, file_path.parent.name
        )
        logger.info(f"📄 Parsing DOCX : {file_path.name}")

        # 3. Extraction du texte structuré + tableaux
        try:
            doc = Document(file_path)
            full_text = self._extract_text_with_structure(doc)
            tables_text = self._extract_tables(doc)
        except Exception as e:
            logger.error(f"Erreur lecture DOCX {file_path.name} : {e}")
            raise

        # 4. Ajout des tableaux en fin de document
        if tables_text:
            full_text += "\n\n" + "=" * 50 + "\n"
            full_text += "DONNÉES TABULAIRES EXTRAITES (Format Markdown)\n"
            full_text += "=" * 50 + "\n"
            full_text += "\n".join(tables_text)

        # 5. Construction du ParsedDocument
        return ParsedDocument(
            source=file_path.stem,
            document_type=self._infer_document_type(file_path.name),
            company=meta["company"],
            period=meta["period"],
            content=full_text.strip(),
            metadata={
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path),
                "total_paragraphs": len(doc.paragraphs),
                "total_tables": len(doc.tables),
            },
            chunks=[],
        )

    # -------------------------------------------------------------------------
    # EXTRACTION DU TEXTE STRUCTURÉ
    # -------------------------------------------------------------------------
    def _extract_text_with_structure(self, doc: Document) -> str:
        """
        Extrait le texte en préservant la hiérarchie des titres.
        Les paragraphes avec un style 'Heading X' deviennent des balises Markdown.
        """
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "").lower()

            # Détection du niveau de titre
            heading_marker = None
            for style_key, marker in self.HEADING_STYLES.items():
                if style_key in style_name:
                    heading_marker = marker
                    break

            if heading_marker:
                parts.append(f"\n{heading_marker} {text}\n")
            else:
                parts.append(text)

        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # EXTRACTION DES TABLEAUX
    # -------------------------------------------------------------------------
    def _extract_tables(self, doc: Document) -> list:
        """Extrait tous les tableaux DOCX en Markdown."""
        tables_text = []
        for i, table in enumerate(doc.tables):
            md = self._table_to_markdown(table)
            if md:
                tables_text.append(f"\n[TABLEAU {i + 1}]\n{md}\n")
        return tables_text

    def _table_to_markdown(self, table: Table) -> str:
        """
        Convertit un tableau python-docx en Markdown.
        Gère les cellules fusionnées en dédupliquant les références XML identiques.
        """
        if not table.rows:
            return ""

        md_lines = []
        for row_idx, row in enumerate(table.rows):
            cells = self._row_to_cells(row)
            if not cells:
                continue

            md_lines.append("| " + " | ".join(cells) + " |")

            # Ligne de séparation après le premier rang (header)
            if row_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def _row_to_cells(row) -> list:
        """
        Convertit une ligne Word en liste de textes.

        Problème résolu : python-docx retourne le MÊME objet cellule plusieurs fois
        quand il y a une fusion horizontale (rowspan). On déduplique via l'identité
        de l'élément XML sous-jacent (`cell._tc`).
        """
        seen_tc_ids = set()
        cells_text = []

        for cell in row.cells:
            tc_id = id(cell._tc)
            if tc_id in seen_tc_ids:
                # Doublon d'une cellule fusionnée → on met une chaîne vide
                cells_text.append("")
            else:
                seen_tc_ids.add(tc_id)
                text = cell.text.strip().replace("\n", " ")
                cells_text.append(text)

        return cells_text

    # -------------------------------------------------------------------------
    # INFÉRENCE DU TYPE DE DOCUMENT
    # -------------------------------------------------------------------------
    def _infer_document_type(self, filename: str) -> str:
        name_lower = filename.lower()

        if (
            "annual_report" in name_lower
            or "annualreport" in name_lower
            or "10k" in name_lower
            or "10-k" in name_lower
        ):
            return "10-K"
        elif (
            "shareholder_letter" in name_lower
            or "shareholderletter" in name_lower
            or "sharholder" in name_lower  # tolérance faute de frappe MSFT
        ):
            return "shareholder_letter"
        elif "proxy" in name_lower:
            return "proxy_statement"
        elif "10-q" in name_lower or "10q" in name_lower:
            return "10-Q"

        return "financial_report"

    def _compute_output_key(self, source_key: str) -> str:
        parts = source_key.split("/")
        filename = parts[-1]
        parent_folder = parts[-2] if len(parts) >= 2 else ""
        meta = self._extract_metadata_from_filename(filename, parent_folder)
        doc_type = self._infer_document_type(filename)
        company_slug = (meta["company"] or "unknown").lower()
        # Slide slug
        import re as _re
        type_slug = _re.sub(r"[^\w]", "-", doc_type.lower())
        stem = Path(filename).stem
        return f"parsed-documents/{company_slug}/{type_slug}/{stem}.json"


# ==============================================================================
# EXÉCUTION DIRECTE (avec idempotence via manifest)
# ==============================================================================
if __name__ == "__main__":
    from src.core.storage import storage
    from src.ingestion.parsers.manifest import load_manifest, save_manifest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = DOCXParser()
    manifest = load_manifest()

    source_keys = [
        k for k in storage.list("raw-documents/presentations/microsoft/")
        if k.lower().endswith(".docx")
    ]
    logger.info(f"🔍 {len(source_keys)} DOCX à parser")

    stats = {"parsed": 0, "skipped": 0, "failed": 0}
    for src in source_keys:
        try:
            doc = parser.parse_and_save_to_storage(src, manifest)
            if doc is None:
                stats["skipped"] += 1
                continue
            logger.info(
                f"✅ [{doc.company}] {src} | Type={doc.document_type} | "
                f"Chars={len(doc.content):,}"
            )
            stats["parsed"] += 1
        except Exception as e:
            logger.error(f"❌ Échec {src} : {e}")
            stats["failed"] += 1

    save_manifest(manifest)
    logger.info(
        f"\n🎉 DOCX : parsed={stats['parsed']} | "
        f"skipped={stats['skipped']} | failed={stats['failed']}"
    )