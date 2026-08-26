import os
import requests
from pathlib import Path

# URLs officielles corrigées (dépôt czyssrs)
FINQA_URLS = {
    "train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
}

def download_finqa_dataset(output_dir: str = "data/raw/finqa") -> None:
    """Télécharge les données de benchmark FinQA dans le répertoire brut."""
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Téléchargement de FinQA dans : {target_path.resolve()}")
    
    for file_name, url in FINQA_URLS.items():
        file_path = target_path / file_name
        
        if file_path.exists():
            print(f"ℹ️ {file_name} existe déjà.")
            continue
            
        print(f"⏳ Téléchargement de {file_name}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ {file_name} sauvegardé avec succès ({len(response.content) / 1024:.1f} KB).")
        except requests.RequestException as e:
            print(f"❌ Erreur sur {file_name}: {e}")

if __name__ == "__main__":
    download_finqa_dataset()