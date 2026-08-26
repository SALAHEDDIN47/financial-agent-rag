# src/ingestion/parsers/json_parser.py
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from src.ingestion.parsers.base_parser import BaseParser, ParsedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class JSONParser(BaseParser):
    """Parser spécialisé pour les fichiers JSON (Transcripts, News, FinQA)."""
    
    def __init__(self):
        super().__init__(document_type="json_data")

    # ✅ FIX 1 : Implémentation de la méthode abstraite requise par BaseParser
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Retourne le premier document parsé. 
        Nécessaire pour respecter le contrat de la classe abstraite BaseParser.
        """
        docs = self.parse_file(file_path)
        if docs:
            return docs[0]
        
        # Fallback si le fichier est vide ou invalide
        return ParsedDocument(
            source=file_path.stem,
            document_type="empty_json",
            company="UNKNOWN",
            period="UNKNOWN",
            content="",
            metadata={"file_path": str(file_path)}
        )

    def parse_file(self, file_path: Path) -> List[ParsedDocument]:
        """
        Parse un fichier JSON et retourne une LISTE de ParsedDocument.
        (Nécessaire car un seul fichier JSON peut contenir des centaines d'articles).
        """
        self._validate_file(file_path, ".json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de lecture JSON dans {file_path.name}: {e}")
            return []

        filename = file_path.name.lower()
        documents = []

        # 1. ROUTING : Détecter le type de fichier
        if "transcript" in filename:
            documents = self._parse_transcript(data, file_path)
        elif "news" in filename:
            documents = self._parse_news(data, file_path)
        elif "finqa" in filename or "dev.json" in filename or "train.json" in filename or "test.json" in filename:
            documents = self._parse_finqa(data, file_path)
        else:
            # Fallback générique
            documents = self._parse_generic(data, file_path)

        logger.info(f"✅ {file_path.name} parsé en {len(documents)} document(s).")
        return documents

    def _parse_transcript(self, data: Dict[str, Any], file_path: Path) -> List[ParsedDocument]:
        """Parse les transcripts d'earnings calls."""
        docs = []
        company = data.get("ticker", self._extract_metadata_from_filename(file_path.name)["company"])
        quarter = data.get("quarter", "UNKNOWN")
        year = data.get("year", "UNKNOWN")
        period = f"Q{quarter}-{year}" if quarter != "UNKNOWN" else "UNKNOWN"
        
        content = data.get("transcript", "")
        if not content and isinstance(data, dict):
            content = max([str(v) for v in data.values() if isinstance(v, str) and len(v) > 1000], key=len, default="")

        if content:
            docs.append(ParsedDocument(
                source=file_path.stem,
                document_type="earnings_transcript",
                company=company.upper(),
                period=period,
                content=content.strip(),
                metadata={
                    "date_published": data.get("datePublished", ""),
                    "file_path": str(file_path)
                },
                chunks=[]
            ))
        return docs

    def _parse_news(self, data: List[Dict[str, Any]], file_path: Path) -> List[ParsedDocument]:
        """Parse les flux d'actualités (liste d'articles)."""
        docs = []
        company = self._extract_metadata_from_filename(file_path.name)["company"]
        
        for idx, article in enumerate(data):
            content_data = article.get("content", {})
            title = content_data.get("title", "Sans titre")
            summary = content_data.get("summary", "")
            pub_date = content_data.get("pubDate", "")
            source = content_data.get("provider", {}).get("displayName", "Unknown")
            url = content_data.get("canonicalUrl", {}).get("url", "")
            
            text_content = f"TITLE: {title}\nDATE: {pub_date}\nSOURCE: {source}\nSUMMARY: {summary}\nURL: {url}"
            
            docs.append(ParsedDocument(
                source=f"{file_path.stem}_article_{idx}",
                document_type="news",
                company=company,
                period="LATEST",
                content=text_content.strip(),
                metadata={
                    "title": title,
                    "url": url,
                    "pub_date": pub_date,
                    "file_path": str(file_path)
                },
                chunks=[]
            ))
        return docs

    def _parse_finqa(self, data: List[Dict[str, Any]], file_path: Path) -> List[ParsedDocument]:
        """Parse le dataset FinQA (Paires QA + Contexte)."""
        docs = []
        max_items = len(data) if len(data) < 5000 else 5000
        logger.info(f"Traitement de {max_items} entrées FinQA (sur {len(data)} au total)...")

        for idx, item in enumerate(data[:max_items]):
            qa = item.get("qa", {})
            table = item.get("table", [])
            pre_text = " ".join(item.get("pre_text", []))
            post_text = " ".join(item.get("post_text", []))
            
            context_text = f"CONTEXT TEXT: {pre_text}\n\n"
            if table:
                table_str = " | ".join([" | ".join(row) for row in table])
                context_text += f"TABLE DATA: {table_str}\n\n"
            context_text += f"POST TEXT: {post_text}"
            
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            explanation = qa.get("explanation", "")
            program = qa.get("program", "")
            
            final_content = (
                f"QUESTION: {question}\n"
                f"ANSWER: {answer}\n"
                f"REASONING: {explanation}\n"
                f"FORMULA: {program}\n\n"
                f"--- SUPPORTING CONTEXT ---\n{context_text}"
            )
            
            company = self._extract_metadata_from_filename(file_path.name)["company"]
            doc_id = item.get("id", f"finqa_{idx}").replace("/", "_")
            
            docs.append(ParsedDocument(
                source=doc_id,
                document_type="finqa_qa_pair",
                company=company,
                period="UNKNOWN",
                content=final_content.strip(),
                metadata={
                    "original_id": item.get("id"),
                    "has_table": len(table) > 0,
                    "file_path": str(file_path)
                },
                chunks=[]
            ))
        return docs

    def _parse_generic(self, data: Any, file_path: Path) -> List[ParsedDocument]:
        """Fallback pour tout autre JSON non identifié."""
        meta = self._extract_metadata_from_filename(file_path.name)
        return [ParsedDocument(
            source=file_path.stem,
            document_type="generic_json",
            company=meta["company"],
            period=meta["period"],
            content=json.dumps(data, indent=2),
            metadata={"file_path": str(file_path)},
            chunks=[]
        )]


# ==============================================================================
# SCRIPT DE TEST / EXÉCUTION DIRECTE
# ==============================================================================
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    parser = JSONParser()
    
    raw_dir = Path("data/raw")
    output_dir = Path("data/interim/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(raw_dir.rglob("*.json"))
    logger.info(f" {len(json_files)} fichiers JSON trouvés au total.")
    
    total_docs_generated = 0
    
    for json_file in json_files:
        try:
            # On utilise parse_file ici car on veut TOUS les documents (ex: tous les articles de news)
            parsed_docs = parser.parse_file(json_file)
            total_docs_generated += len(parsed_docs)
            
            for doc in parsed_docs:
                company_dir = output_dir / doc.company.lower()
                company_dir.mkdir(parents=True, exist_ok=True)
                
                safe_source = re.sub(r'[^\w\-_\. ]', '_', doc.source)
                output_file = company_dir / f"{safe_source}.json"
                
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
            logger.error(f"❌ Échec du parsing de {json_file.name} : {e}")
            
    logger.info(f"\n{'='*60}")
    logger.info(f" Parsing JSON terminé ! {total_docs_generated} documents générés au total.")
    logger.info(f" Sauvegardés dans : {output_dir.absolute()}")
    logger.info(f"{'='*60}")