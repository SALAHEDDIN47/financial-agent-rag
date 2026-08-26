# src/ingestion/parsers/base_parser.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """Format standardisé de sortie pour TOUS les parsers."""
    source: str
    document_type: str
    company: str
    period: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)

class BaseParser(ABC):
    """Classe abstraite que tous les parsers spécifiques doivent hériter."""
    
    def __init__(self, document_type: str):
        self.document_type = document_type

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        pass

    def _extract_metadata_from_filename(self, filename: str, parent_folder: str = "") -> Dict[str, str]:
        """
        Extrait company et period du nom de fichier ET du dossier parent.
        parent_folder est crucial pour les fichiers comme '2024-annual-report.pdf' 
        qui sont dans le dossier 'alphabet'.
        """
        name = filename.lower()
        folder = parent_folder.lower()
        
        company = "UNKNOWN"
        
        # 1. Détection via le dossier parent (le plus fiable pour Microsoft/Alphabet)
        if "tesla" in folder: company = "TSLA"
        elif "microsoft" in folder or "msft" in folder: company = "MSFT"
        elif "alphabet" in folder or "google" in folder or "goog" in folder: company = "GOOGL"
        elif "apple" in folder or "aapl" in folder: company = "AAPL"
        
        # 2. Détection via le nom de fichier (fallback)
        if company == "UNKNOWN":
            if "tsla" in name: company = "TSLA"
            elif "msft" in name or "microsoft" in name: company = "MSFT"
            elif "goog" in name or "alphabet" in name: company = "GOOGL"
            elif "aapl" in name or "apple" in name: company = "AAPL"
            
        # 3. Extraction de la période
        period = "UNKNOWN"
        # Patterns : Q1-2024, 2024q1, FY2024, FY24, Annual Report 2024
        period_match = re.search(r'(q[1-4]-?\d{4}|\d{4}q[1-4]|fy\d{4}|fy\d{2})', name)
        if period_match:
            period = period_match.group(1).upper().replace('-', '')
        else:
            # Fallback pour les rapports annuels : "2024_Annual_Report.pdf"
            year_match = re.search(r'(20\d{2})', name)
            if year_match:
                period = f"FY{year_match.group(1)}"
            
        return {"company": company, "period": period}

    def _validate_file(self, file_path: Path, expected_extension: str) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas : {file_path}")
        if file_path.suffix.lower() != expected_extension:
            raise ValueError(f"Extension attendue : {expected_extension}, reçue : {file_path.suffix}")