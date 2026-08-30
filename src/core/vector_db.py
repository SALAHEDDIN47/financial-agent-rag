from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_milvus_collection(collection_name: str = "financial_chunks", dim: int = 1024):
    """Crée la collection Milvus avec le schéma approprié."""
    connections.connect("default", host="localhost", port="19530")
    
    if utility.has_collection(collection_name):
        logger.info(f"⚠️ La collection '{collection_name}' existe déjà. Suppression...")
        utility.drop_collection(collection_name)
        
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="company", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="period", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields, description="Financial RAG Chunks Collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Index pour la similarité cosinus
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info(f"✅ Collection '{collection_name}' créée et indexée.")
    return collection

def ingest_to_milvus(collection_name: str, input_dir: str, batch_size: int = 500):
    """Lit les fichiers JSONL et les insère dans Milvus par lots."""
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(collection_name)
    
    input_path = Path(input_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))
    logger.info(f"📂 {len(jsonl_files)} fichiers JSONL à ingérer dans Milvus.")
    
    total_inserted = 0
    for file in jsonl_files:
        logger.info(f"📥 Insertion des données de {file.name}...")
        entities = {
            "chunk_id": [],
            "company": [],
            "period": [],
            "document_type": [],
            "text": [],
            "embedding": []
        }
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                entities["chunk_id"].append(str(chunk.get("chunk_id", "")))
                entities["company"].append(str(chunk.get("company", "UNKNOWN")))
                entities["period"].append(str(chunk.get("period", "UNKNOWN")))
                entities["document_type"].append(str(chunk.get("document_type", "UNKNOWN")))
                # Limite de sécurité pour le champ VARCHAR de Milvus
                entities["text"].append(str(chunk.get("text", ""))[:3900]) 
                entities["embedding"].append(chunk.get("embedding", []))
                
                # Insertion par lots pour optimiser la mémoire et la vitesse
                if len(entities["text"]) >= batch_size:
                    # ✅ FIX CRITIQUE : Convertir le dictionnaire de listes en liste de dictionnaires (rows)
                    rows = [
                        {
                            "chunk_id": entities["chunk_id"][i],
                            "company": entities["company"][i],
                            "period": entities["period"][i],
                            "document_type": entities["document_type"][i],
                            "text": entities["text"][i],
                            "embedding": entities["embedding"][i]
                        }
                        for i in range(len(entities["chunk_id"]))
                    ]
                    collection.insert(rows)
                    total_inserted += len(entities["text"])
                    logger.info(f"  -> {total_inserted} chunks insérés...")
                    
                    # Reset des listes
                    for key in entities:
                        entities[key] = []
                        
        # Insertion du reste des données (dernier batch)
        if len(entities["text"]) > 0:
            rows = [
                {
                    "chunk_id": entities["chunk_id"][i],
                    "company": entities["company"][i],
                    "period": entities["period"][i],
                    "document_type": entities["document_type"][i],
                    "text": entities["text"][i],
                    "embedding": entities["embedding"][i]
                }
                for i in range(len(entities["chunk_id"]))
            ]
            collection.insert(rows)
            total_inserted += len(entities["text"])
            logger.info(f"  -> {total_inserted} chunks insérés au total pour ce fichier.")
            
    # Charger la collection en mémoire pour la recherche
    collection.load()
    logger.info(f"🎉 Ingestion terminée ! {total_inserted} chunks au total dans Milvus (Collection chargée).")

if __name__ == "__main__":
    # 1. Créer la collection et l'index
    setup_milvus_collection("financial_chunks", dim=1024)
    
    # 2. Ingérer les données embeddées
    ingest_to_milvus(
        collection_name="financial_chunks",
        input_dir="data/processed/embedded_chunks",
        batch_size=500
    )