# src/ingestion/embedder.py
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_embeddings(input_dir: str, output_dir: str, model_name: str = "BAAI/bge-large-en-v1.5"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🧠 Chargement du modèle d'embedding : {model_name}")
    # normalize_embeddings=True est crucial pour la similarité cosinus dans Milvus
    model = SentenceTransformer(model_name)
    
    jsonl_files = list(input_path.glob("*.jsonl"))
    logger.info(f"📂 Traitement de {len(jsonl_files)} fichiers de chunks...")
    
    for file in jsonl_files:
        logger.info(f"⏳ Traitement de {file.name}...")
        output_file = output_path / file.name
        if output_file.exists():
            print(f"⏭️ Déjà traité, on passe : {file.name}")
            continue
        
        with open(file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                chunk = json.loads(line)
                text = chunk.get("text", "")
                
                # Génération de l'embedding (vecteur de 1024 dimensions)
                embedding = model.encode(text, normalize_embeddings=True).tolist()
                chunk["embedding"] = embedding
                
                f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                
        logger.info(f"✅ Terminé pour {file.name}")

if __name__ == "__main__":
    generate_embeddings(
        input_dir="data/processed/chunks",
        output_dir="data/processed/embedded_chunks",
        model_name="BAAI/bge-large-en-v1.5"
    )