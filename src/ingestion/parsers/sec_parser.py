# src/ingestion/parsers/sec_parser.py
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SECParser(BaseParser):
    """
    Parser spécialisé pour les fichiers 'full-submission.txt' de la SEC EDGAR.
    Il extrait le texte lisible du dépôt principal (10-K, 10-Q, 8-K) en ignorant 
    les données XBRL brutes et les pièces jointes techniques.
    """
    
    def __init__(self):
        super().__init__(document_type="sec_filing")

    def parse(self, file_path: Path) -> ParsedDocument:
        # 1. Validation
        self._validate_file(file_path, ".txt")
        
        # 2. Extraction des métadonnées depuis le chemin et le fichier
        meta = self._extract_metadata_from_filename(file_path.name, file_path.parent.parent.parent.name)
        sec_meta = self._parse_sec_header(file_path)
        
        logger.info(f"📄 Parsing SEC Filing : {file_path.name} ({sec_meta.get('type', 'Unknown')})")
        
        # 3. Lecture du fichier
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()
            
        # 4. Extraction du texte principal
        clean_text = self._extract_main_text(raw_content)
        
        if not clean_text or len(clean_text) < 100:
            logger.warning(f"⚠️ Très peu de texte extrait pour {file_path.name}. Le document est peut-être uniquement en XBRL.")
            
        # 5. Construction de l'objet final
        return ParsedDocument(
            source=file_path.stem,
            document_type=sec_meta.get("type", "sec_filing"),
            company=meta["company"],
            period=sec_meta.get("period", meta["period"]),
            content=clean_text.strip(),
            metadata={
                "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                "file_path": str(file_path),
                "sec_accession": file_path.parent.name,
                "filing_date": sec_meta.get("filing_date", "")
            },
            chunks=[]
        )

    def _parse_sec_header(self, file_path: Path) -> dict:
        """Extrait les métadonnées spécifiques à la SEC depuis l'en-tête du fichier."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # On lit seulement les 50 premières lignes pour trouver l'en-tête rapidement
            header_text = "".join([next(f) for _ in range(50)])
            
        doc_type = "Unknown"
        period = "Unknown"
        filing_date = ""
        
        type_match = re.search(r'CONFORMED SUBMISSION TYPE:\s*(.*)', header_text)
        if type_match:
            doc_type = type_match.group(1).strip()
            
        period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', header_text)
        if period_match:
            date_str = period_match.group(1)
            period = f"FY{date_str[:4]}" # Ex: FY2024
            
        date_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', header_text)
        if date_match:
            d = date_match.group(1)
            filing_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            
        return {"type": doc_type, "period": period, "filing_date": filing_date}

    def _extract_main_text(self, raw_content: str) -> str:
        """Trouve le premier document principal (10-K, 10-Q, 8-K) et nettoie son HTML."""
        # On divise le fichier en blocs <DOCUMENT>
        documents = re.split(r'<DOCUMENT>', raw_content)
        
        types_to_keep = ['10-K', '10-Q', '8-K', '20-F', '6-K', 'SC 13G', 'SC 13D']
        
        for doc in documents[1:]: # On ignore tout ce qu'il y a avant le premier <DOCUMENT>
            type_match = re.search(r'<TYPE>(.*)', doc)
            if type_match:
                doc_type = type_match.group(1).strip().upper()
                
                # On cherche le rapport principal (pas les pièces jointes EX- ni le XBRL pur)
                if doc_type in types_to_keep:
                    text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
                    if text_match:
                        inner_text = text_match.group(1)
                        
                        # Si c'est du HTML, on utilise BeautifulSoup pour extraire le texte proprement
                        if '<html' in inner_text.lower() or '<body' in inner_text.lower():
                            soup = BeautifulSoup(inner_text, 'lxml')
                            # On supprime les scripts et styles
                            for script in soup(["script", "style"]):
                                script.decompose()
                            clean_text = soup.get_text(separator='\n')
                        else:
                            # Fallback si c'est du texte brut ou XML
                            clean_text = inner_text
                            
                        # Nettoyage des espaces et sauts de ligne excessifs
                        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
                        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
                        return clean_text
                        
        # Fallback : si aucun type principal n'est trouvé, on prend le premier <TEXT> disponible
        first_text = re.search(r'<TEXT>(.*?)</TEXT>', raw_content, re.DOTALL)
        if first_text:
            soup = BeautifulSoup(first_text.group(1), 'lxml')
            return soup.get_text(separator='\n')
            
        return ""


# ==============================================================================
# SCRIPT DE TEST / EXÉCUTION DIRECTE
# ==============================================================================
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    parser = SECParser()
    
    # Cible : le dossier SEC
    target_dir = Path("data/raw/sec/sec-edgar-filings")
    output_dir = Path("data/interim/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not target_dir.exists():
        logger.error(f"Le dossier {target_dir} n'existe pas.")
    else:
        # Trouver tous les full-submission.txt
        txt_files = list(target_dir.rglob("full-submission.txt"))
        logger.info(f"🔍 {len(txt_files)} fichiers SEC trouvés.")
        
        for txt_file in txt_files:
            try:
                doc = parser.parse(txt_file)
                
                # Création d'un sous-dossier par entreprise
                company_dir = output_dir / doc.company.lower() / "sec"
                company_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"✅ [{doc.company}] {doc.source} | Type: {doc.document_type} | "
                            f"Période: {doc.period} | Caractères: {len(doc.content)}")
                
                import json
                output_file = company_dir / f"{doc.source}.json"
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
                logger.error(f"❌ Échec du parsing de {txt_file.name} : {e}")
                
        logger.info(f"🎉 Parsing SEC terminé ! Fichiers sauvegardés dans : {output_dir.absolute()}")