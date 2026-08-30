# src/ingestion/chunking/financial_chunker.py
import json
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def chunk_documents(input_dir: Path, output_dir: Path, chunk_size: int = 1200, chunk_overlap: int = 150):
    """
    Découpe tous les documents JSON parsés en chunks et les sauvegarde en JSONL.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Le RecursiveCharacterTextSplitter est le meilleur pour du texte financier.
    # Il essaie de couper aux doubles sauts de ligne, puis aux simples, puis aux points, etc.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    # Trouve tous les fichiers JSON récursivement (dans aapl/, msft/, tsla/, etc.)
    json_files = list(input_dir.rglob("*.json"))
    logger.info(f" {len(json_files)} fichiers JSON à chunker dans {input_dir}")
    
    total_chunks = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                
            content = doc.get("content", "")
            if not content or len(content) < 100:
                continue
                
            # Découpage du texte
            chunks = splitter.split_text(content)
            
            # Métadonnées de base à attacher à chaque chunk
            base_metadata = {
                "source_file": json_file.name,
                "source": doc.get("source", "unknown"),
                "company": doc.get("company", "unknown"),
                "document_type": doc.get("document_type", "unknown"),
                "period": doc.get("period", "unknown"),
                "file_path": doc.get("metadata", {}).get("file_path", "")
            }
            
            # Sauvegarde en JSONL (une ligne JSON par chunk)
            output_file = output_dir / f"{json_file.stem}_chunks.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for i, chunk_text in enumerate(chunks):
                    chunk_data = {
                        "chunk_id": f"{doc.get('source', 'doc')}_{i:04d}",
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    }
                    f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                    
            total_chunks += len(chunks)
            logger.info(f"✅ {json_file.name} découpé en {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chunking de {json_file.name} : {e}")
            
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Chunking terminé ! {total_chunks} chunks générés au total.")
    logger.info(f"📁 Sauvegardés dans : {output_dir.absolute()}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    input_dir = Path("data/interim/parsed")
    output_dir = Path("data/processed/chunks")
    
    # Note : Pour les 10-K de 400k+ caractères, 1200 chars est un bon compromis 
    # pour garder du contexte sans faire des chunks trop petits.
    chunk_documents(input_dir, output_dir, chunk_size=1200, chunk_overlap=150)