# src/ingestion/parsers/base_parser.py
"""
BaseParser : contrat commun à tous les parsers + logique d'idempotence.

Fonctionnalités clés :
  - ParsedDocument : format de sortie standardisé
  - compute_file_hash() : détection de vraie modification
  - needs_parsing() : décision "parser ou ignorer"
  - BaseParser.parse_and_save() : wrapper idempotent (parse → save → manifest)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# MODÈLE DE DONNÉES
# ==============================================================================
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


# ==============================================================================
# UTILITAIRES D'IDEMPOTENCE
# ==============================================================================
def compute_file_hash(file_path: Path, algo: str = "md5", chunk_size: int = 8192) -> str:
    """
    Calcule le hash du contenu d'un fichier.
    Utilisé pour détecter les vraies modifications (vs. simple changement de mtime).
    """
    h = hashlib.new(algo)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def needs_parsing(source: Path, output: Path, manifest: dict) -> bool:
    """
    Détermine si `source` doit être (re)parsé vers `output`.

    Retourne True si :
      - le fichier de sortie n'existe pas, OU
      - le fichier source n'est pas dans le manifest, OU
      - le hash du fichier source a changé depuis le dernier parsing.

    Retourne False sinon (→ on peut ignorer le parsing).
    """
    if not output.exists():
        return True

    key = str(source)
    prev = manifest.get(key)
    if not prev:
        return True

    try:
        current_hash = compute_file_hash(source)
    except (FileNotFoundError, PermissionError):
        # Fichier source disparu ou inaccessible → on considère qu'il faut re-parser
        return True

    return current_hash != prev.get("source_hash")


# ==============================================================================
# CLASSE ABSTRAITE
# ==============================================================================
class BaseParser(ABC):
    """Classe abstraite que tous les parsers spécifiques doivent hériter."""

    def __init__(self, document_type: str):
        self.document_type = document_type

    # -------------------------------------------------------------------------
    # CONTRAT : à implémenter dans chaque sous-classe
    # -------------------------------------------------------------------------
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse un fichier et retourne un ParsedDocument."""
        pass

    # -------------------------------------------------------------------------
    # MÉTADONNÉES
    # -------------------------------------------------------------------------
    def _extract_metadata_from_filename(
        self, filename: str, parent_folder: str = ""
    ) -> Dict[str, str]:
        """
        Extrait company et period depuis le nom de fichier + dossier parent.
        Le dossier parent est crucial pour les fichiers comme '2024-annual-report.pdf'
        qui sont dans le dossier 'alphabet'.
        """
        name = filename.lower()
        folder = parent_folder.lower()

        company = "UNKNOWN"

        # 1. Détection via le dossier parent (le plus fiable)
        # ✅ FIX : ajout de "tsla" pour matcher le dossier SEC (data/raw/sec/.../TSLA/)
        if "tesla" in folder or "tsla" in folder:
            company = "TSLA"
        elif "microsoft" in folder or "msft" in folder:
            company = "MSFT"
        elif "alphabet" in folder or "google" in folder or "goog" in folder:
            company = "GOOGL"
        elif "apple" in folder or "aapl" in folder:
            company = "AAPL"

        # 2. Détection via le nom de fichier (fallback)
        if company == "UNKNOWN":
            if "tsla" in name or "tesla" in name:
                company = "TSLA"
            elif "msft" in name or "microsoft" in name:
                company = "MSFT"
            elif "goog" in name or "alphabet" in name:
                company = "GOOGL"
            elif "aapl" in name or "apple" in name:
                company = "AAPL"

        # 3. Extraction de la période
        period = "UNKNOWN"
        period_match = re.search(
            r"(q[1-4]-?\d{4}|\d{4}q[1-4]|fy\d{4}|fy\d{2})", name
        )
        if period_match:
            period = period_match.group(1).upper().replace("-", "")
        else:
            # Fallback pour les rapports annuels : "2024_Annual_Report.pdf"
            year_match = re.search(r"(20\d{2})", name)
            if year_match:
                period = f"FY{year_match.group(1)}"

        return {"company": company, "period": period}

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    def _validate_file(self, file_path: Path, expected_extension: str) -> None:
        """Vérifie que le fichier existe et a la bonne extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas : {file_path}")
        if file_path.suffix.lower() != expected_extension:
            raise ValueError(
                f"Extension attendue : {expected_extension}, reçue : {file_path.suffix}"
            )

    # -------------------------------------------------------------------------
    # SAUVEGARDE (JSON)
    # -------------------------------------------------------------------------
    def save_document(self, doc: ParsedDocument, output_file: Path) -> None:
        """
        Sérialise un ParsedDocument en JSON.
        Centralisé ici pour garantir un format identique pour tous les parsers.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": doc.source,
                    "document_type": doc.document_type,
                    "company": doc.company,
                    "period": doc.period,
                    "content": doc.content,
                    "metadata": doc.metadata,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # -------------------------------------------------------------------------
    # WRAPPER IDEMPOTENT
    # -------------------------------------------------------------------------
    def parse_and_save(
        self,
        source: Path,
        output_file: Path,
        manifest: dict,
        force: bool = False,
    ) -> Optional[ParsedDocument]:
        """
        Wrapper idempotent : parse `source` UNIQUEMENT si nécessaire,
        sauvegarde le JSON, met à jour le manifest.

        Paramètres
        ----------
        source : Path
            Fichier source à parser.
        output_file : Path
            Fichier JSON de sortie.
        manifest : dict
            Manifest mutable chargé via `load_manifest()`. Mis à jour en place.
        force : bool
            Si True, force le reparsing même si déjà à jour.

        Retourne
        --------
        ParsedDocument si parsé, None si ignoré (déjà à jour).
        """
        # 1. Vérification idempotente
        if not force and not needs_parsing(source, output_file, manifest):
            logger.debug(f"⏭️  Ignoré (déjà parsé) : {source.name}")
            return None

        # 2. Parsing effectif
        doc = self.parse(source)

        # 3. Enrichissement des métadonnées pour traçabilité
        source_hash = compute_file_hash(source)
        parsed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        doc.metadata["source_hash"] = source_hash
        doc.metadata["parsed_at"] = parsed_at

        # 4. Sauvegarde
        self.save_document(doc, output_file)

        # 5. Mise à jour du manifest (mutation en place)
        manifest[str(source)] = {
            "source_hash": source_hash,
            "parsed_at": parsed_at,
            "output": str(output_file),
            "document_type": doc.document_type,
        }

        return doc