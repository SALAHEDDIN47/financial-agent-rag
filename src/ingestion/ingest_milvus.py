# src/ingestion/ingest_milvus.py
import json
import logging
from pathlib import Path
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_milvus_collection(collection_name: str = "financial_chunks", dim: int = 1024):
    """Se connecte à Milvus et crée la collection si elle n'existe pas."""
    # Connexion à Milvus (ports par défaut du docker-compose)
    connections.connect("default", host="localhost", port="19530")
    logger.info("✅ Connecté à Milvus")

    if utility.has_collection(collection_name):
        logger.info(f"⚠️ La collection '{collection_name}' existe déjà. Elle sera supprimée pour une réinitialisation propre.")
        utility.drop_collection(collection_name)

    # Définition du schéma de la collection
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="period", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # Max supporté par Milvus 2.3+
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields, description="Collection pour les chunks financiers RAG")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"✅ Collection '{collection_name}' créée avec succès.")

    # Création de l'index pour la recherche vectorielle (HNSW est très performant pour ce cas d'usage)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("✅ Index vectoriel (HNSW) créé sur le champ 'embedding'.")
    
    return collection

def ingest_data_to_milvus(collection_name: str = "financial_chunks", input_dir: str = "data/processed/embedded_chunks", batch_size: int = 500):
    """Lit les fichiers JSONL embeddés et les insère dans Milvus par lots."""
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(collection_name)
    
    input_path = Path(input_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        logger.error(f"❌ Aucun fichier JSONL trouvé dans {input_dir}. Avez-vous bien lancé le script d'embedding au préalable ?")
        return

    logger.info(f"📂 Début de l'ingestion de {len(jsonl_files)} fichiers JSONL dans Milvus...")
    
    total_inserted = 0
    for file in jsonl_files:
        logger.info(f"⏳ Traitement de {file.name}...")
        batch_data = []
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                
                # Préparation des données pour l'insertion (format liste de dictionnaires requis par PyMilvus)
                batch_data.append({
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "company": str(chunk.get("company", "UNKNOWN")),
                    "period": str(chunk.get("period", "UNKNOWN")),
                    "document_type": str(chunk.get("document_type", "UNKNOWN")),
                    "source": str(chunk.get("source", "")),
                    "text": str(chunk.get("text", ""))[:65000], # Sécurité pour la limite VARCHAR de Milvus
                    "embedding": chunk.get("embedding", [])
                })
                
                # Insertion par lot pour optimiser la mémoire et la vitesse
                if len(batch_data) >= batch_size:
                    collection.insert(batch_data)
                    total_inserted += len(batch_data)
                    logger.info(f"  -> {total_inserted} chunks insérés au total...")
                    batch_data = [] # Réinitialisation du batch
            
            # Insertion du dernier batch s'il reste des données
            if batch_data:
                collection.insert(batch_data)
                total_inserted += len(batch_data)
                logger.info(f"  -> {total_inserted} chunks insérés au total pour ce fichier.")
                
    # Chargement de la collection en mémoire pour permettre la recherche immédiate
    collection.load()
    logger.info(f"🎉 Ingestion terminée ! {total_inserted} chunks au total dans la collection '{collection_name}'.")
    logger.info("✅ La collection est chargée en mémoire et prête pour la recherche.")

if __name__ == "__main__":
    # 1. Créer la collection et l'index
    setup_milvus_collection(collection_name="financial_chunks", dim=1024)
    
    # 2. Ingérer les données embeddées
    ingest_data_to_milvus(
        collection_name="financial_chunks",
        input_dir="data/processed/embedded_chunks",
        batch_size=500
    )