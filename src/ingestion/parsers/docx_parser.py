# src/ingestion/parsers/docx_parser.py
import logging
from pathlib import Path
from docx import Document

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)

class DOCXParser(BaseParser):
    """Parser spécialisé pour les rapports annuels et lettres aux actionnaires au format Word (.docx)"""
    
    def __init__(self):
        super().__init__(document_type="docx_report")

    def parse(self, file_path: Path) -> ParsedDocument:
        # 1. Validation
        self._validate_file(file_path, ".docx")
        
        # 2. Métadonnées
        meta = self._extract_metadata_from_filename(file_path.name, file_path.parent.name)
        logger.info(f"📄 Parsing DOCX en cours : {file_path.name}")
        
        # 3. Extraction du texte
        try:
            doc = Document(file_path)
            # On extrait tous les paragraphes non vides
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs)
            
            # Optionnel : Extraction basique des tableaux dans le Word
            tables_text = []
            for i, table in enumerate(doc.tables):
                table_md = self._table_to_markdown(table)
                tables_text.append(f"\n[TABLEAU {i+1}]\n{table_md}\n")
            
            if tables_text:
                full_text += "\n\n" + "="*50 + "\n"
                full_text += "DONNÉES TABULAIRES EXTRAITES (Format Markdown)\n"
                full_text += "="*50 + "\n"
                full_text += "\n".join(tables_text)

        except Exception as e:
            logger.error(f"Erreur lors de la lecture du DOCX {file_path.name}: {e}")
            raise

        # 4. Retour de l'objet standardisé
        return ParsedDocument(
            source=file_path.stem,
            document_type=self._infer_document_type(file_path.name),
            company=meta["company"],
            period=meta["period"],
            content=full_text.strip(),
            metadata={
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path)
            },
            chunks=[]
        )

    def _table_to_markdown(self, table) -> str:
        """Convertit un tableau python-docx en Markdown"""
        md_lines = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(md_lines)

    def _infer_document_type(self, filename: str) -> str:
        name_lower = filename.lower()
        if "annual_report" in name_lower or "10k" in name_lower:
            return "10-K"
        elif "shareholder_letter" in name_lower:
            return "shareholder_letter"
        elif "proxy" in name_lower:
            return "proxy_statement"
        return "financial_report"


# ==============================================================================
# TEST DIRECT
# ==============================================================================
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = DOCXParser()
    
    target_dir = Path("data/raw/presentations/microsoft")
    output_dir = Path("data/interim/parsed/msft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    docx_files = list(target_dir.glob("*.docx"))
    logger.info(f"🔍 {len(docx_files)} fichiers DOCX trouvés")
    
    for docx_file in docx_files:
        try:
            doc = parser.parse(docx_file)
            logger.info(f"✅ [{doc.company}] {doc.source} | Type: {doc.document_type} | Caractères: {len(doc.content)}")
            
            import json
            output_file = output_dir / f"{doc.source}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "source": doc.source,
                    "document_type": doc.document_type,
                    "company": doc.company,
                    "period": doc.period,
                    "content": doc.content,
                    "metadata": doc.metadata
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Échec : {docx_file.name} -> {e}")