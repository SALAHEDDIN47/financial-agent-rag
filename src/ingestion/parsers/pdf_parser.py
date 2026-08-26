# src/ingestion/parsers/pdf_parser.py
import sys
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PDFParser(BaseParser):
    def __init__(self, preserve_tables: bool = True):
        super().__init__(document_type="pdf_document")
        self.preserve_tables = preserve_tables

    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path, ".pdf")
        
        # ✅ NOUVEAU : On passe le nom du dossier parent pour mieux détecter l'entreprise
        meta = self._extract_metadata_from_filename(file_path.name, file_path.parent.name)
        
        logger.info(f"📄 Parsing en cours : {file_path.name} (Dossier: {file_path.parent.name})")
        
        full_text = ""
        tables_data = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = re.sub(r'\n{3,}', '\n\n', text).strip()
                
                if text:
                    full_text += f"\n\n--- PAGE {page.page_number} ---\n{text}"
                
                if self.preserve_tables:
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            table_md = self._table_to_markdown(table)
                            tables_data.append({
                                "page": page.page_number,
                                "table_idx": table_idx,
                                "content": table_md
                            })
        
        if tables_data:
            full_text += "\n\n" + "="*50 + "\n"
            full_text += "DONNÉES TABULAIRES EXTRAITES (Format Markdown)\n"
            full_text += "="*50 + "\n"
            for tbl in tables_data:
                full_text += f"\n[Source: Page {tbl['page']}, Tableau {tbl['table_idx']}]\n"
                full_text += tbl['content'] + "\n"
        
        doc_type = self._infer_document_type(file_path.name)
        
        return ParsedDocument(
            source=file_path.stem,
            document_type=doc_type,
            company=meta["company"],
            period=meta["period"],
            content=full_text.strip(),
            metadata={
                "total_pages": len(pdf.pages),
                "tables_found": len(tables_data),
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path)
            },
            chunks=[]
        )

    def _table_to_markdown(self, table: List[List[Any]]) -> str:
        if not table:
            return ""
        cleaned_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
        max_cols = max(len(row) for row in cleaned_table)
        for row in cleaned_table:
            row.extend([""] * (max_cols - len(row)))
        
        header = cleaned_table[0]
        separator = ["---"] * max_cols
        rows = cleaned_table[1:]
        
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |"
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)

    def _infer_document_type(self, filename: str) -> str:
        name_lower = filename.lower()
        if "10-k" in name_lower or "annual_report" in name_lower or "annualreport" in name_lower:
            return "10-K"
        elif "10-q" in name_lower or "earnings_release" in name_lower or "update" in name_lower:
            return "10-Q"
        elif "slides" in name_lower or "presentation" in name_lower or "deck" in name_lower:
            return "presentation_slides"
        elif "proxy" in name_lower:
            return "proxy_statement"
        elif "shareholder_letter" in name_lower or "shareholderletter" in name_lower:
            return "shareholder_letter"
        else:
            return "financial_report"


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    parser = PDFParser(preserve_tables=True)
    
    # ✅ NOUVEAU : On cible le dossier parent qui contient tesla/, microsoft/, alphabet/
    target_dir = Path("data/raw/presentations")
    output_dir = Path("data/interim/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not target_dir.exists():
        logger.error(f"Le dossier {target_dir} n'existe pas.")
    else:
        # ✅ NOUVEAU : rglob pour trouver les PDF dans tous les sous-dossiers
        pdf_files = list(target_dir.rglob("*.pdf"))
        logger.info(f"🔍 {len(pdf_files)} fichiers PDF trouvés au total dans {target_dir}")
        
        # Statistiques par entreprise
        stats = {}
        
        for pdf_file in pdf_files:
            try:
                doc = parser.parse(pdf_file)
                
                # Création d'un sous-dossier de sortie par entreprise pour bien organiser
                company_output = output_dir / doc.company.lower()
                company_output.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"✅ [{doc.company}] {doc.source} | Type: {doc.document_type} | "
                            f"Pages: {doc.metadata['total_pages']} | Tables: {doc.metadata['tables_found']}")
                
                output_file = company_output / f"{doc.source}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "source": doc.source,
                        "document_type": doc.document_type,
                        "company": doc.company,
                        "period": doc.period,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }, f, indent=2, ensure_ascii=False)
                
                # Mise à jour des stats
                stats[doc.company] = stats.get(doc.company, 0) + 1
                    
            except Exception as e:
                logger.error(f"❌ Échec du parsing de {pdf_file.name} : {e}")
                
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Parsing terminé ! Résumé par entreprise :")
        for company, count in stats.items():
            logger.info(f"   📊 {company}: {count} fichiers parsés")
        logger.info(f"📁 Fichiers sauvegardés dans : {output_dir.absolute()}")
        logger.info(f"{'='*60}")