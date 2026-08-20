"""
Collecteur de rapports 10-K depuis la SEC EDGAR.
Télécharge les PDFs et les range dans data/raw/{CompanyName}/10K_YYYY.pdf
Gère correctement les années 19xx et 20xx.
"""

import requests
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# CIK des entreprises (Central Index Key)
COMPANIES = {
    "Tesla": "0001318605",
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    # Vous pouvez en ajouter d'autres ici
    # "Amazon": "0001018724",
    # "Google": "0001652044",
}

# User-Agent OBLIGATOIRE pour SEC EDGAR
# ⚠️ Remplacez par vos vraies informations (nom, email)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YourName/1.0; your.email@domain.com)"
}

# ==================== FONCTIONS UTILITAIRES ====================

def extract_year_from_accession(accession_number: str) -> Optional[str]:
    """
    Extrait l'année réelle du numéro d'accession SEC.
    Gère les années 1990-2099.
    
    Exemples :
        "0001234567-23-000001" -> "2023"
        "0001234567-98-000001" -> "1998"
        "0001234567-05-000001" -> "2005"
    
    Règle : si le suffixe (deux derniers chiffres) est >= 50, on suppose 19xx,
    sinon 20xx. Cette heuristique est valable pour les documents du 20e/21e siècle.
    """
    parts = accession_number.split('-')
    if len(parts) >= 2:
        year_suffix = parts[1]  # ex: "23", "98", "05"
        if year_suffix.isdigit() and len(year_suffix) == 2:
            suffix_int = int(year_suffix)
            # Règle du siècle : >= 50 → 19xx, sinon 20xx
            # Cela couvre 1950-1999 et 2000-2049 (et jusqu'à 2099 pour 99)
            if suffix_int >= 50:
                return f"19{year_suffix}"
            else:
                return f"20{year_suffix}"
    return None


def get_filings(cik: str, count: int = 3) -> List[dict]:
    """
    Récupère les derniers dépôts 10-K via l'API SEC.
    Retourne une liste de dictionnaires contenant l'accession, le document principal et l'URL.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        filings = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for form, acc, doc in zip(forms, accessions, primary_docs):
            if form == "10-K":  # On ignore les amendements (10-K/A)
                filings.append({
                    "accession": acc,
                    "primary_doc": doc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
                })
                if len(filings) >= count:
                    break
        return filings

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur réseau pour CIK {cik}: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue pour CIK {cik}: {e}")
    return []


def download_pdfs():
    """
    Télécharge les PDFs des 10-K et les range dans :
    data/raw/NomEntreprise/10K_YYYY.pdf
    """
    for name, cik in COMPANIES.items():
        print(f"\n🔍 Récupération des 10-K pour {name} (CIK: {cik})...")

        # Création du sous-dossier dédié à l'entreprise
        company_dir = RAW_DATA_DIR / name
        company_dir.mkdir(parents=True, exist_ok=True)

        # Récupération des 2 derniers 10-K (généralement N-1 et N-2)
        filings = get_filings(cik, count=2)

        if not filings:
            print(f"⚠️ Aucun 10-K trouvé pour {name}.")
            continue

        for filing in filings:
            # Extraction correcte de l'année
            year = extract_year_from_accession(filing["accession"])
            if year is None:
                print(f"⚠️ Année non reconnue pour {filing['accession']}, utilisation du numéro d'accession complet.")
                safe_name = filing["accession"].replace('-', '')
                file_path = company_dir / f"10K_{safe_name}.pdf"
            else:
                file_path = company_dir / f"10K_{year}.pdf"

            # Vérification si le fichier existe déjà
            if file_path.exists():
                print(f"⏩ {file_path.name} existe déjà, ignoré.")
                continue

            print(f"⬇️ Téléchargement de {file_path.name}...")
            try:
                pdf_resp = requests.get(filing["url"], headers=HEADERS, timeout=60)
                pdf_resp.raise_for_status()

                # Sauvegarde du PDF
                with open(file_path, "wb") as f:
                    f.write(pdf_resp.content)

                print(f"✅ Sauvegardé : {file_path}")
                time.sleep(1.5)  # Politesse envers le serveur SEC (rate limiting)

            except requests.exceptions.RequestException as e:
                print(f"❌ Erreur réseau pour {file_path.name}: {e}")
            except Exception as e:
                print(f"❌ Erreur inattendue pour {file_path.name}: {e}")

        # Pause supplémentaire entre deux entreprises
        time.sleep(2)

    print("\n🎯 Collecte terminée !")


# ==================== POINT D'ENTRÉE ====================
if __name__ == "__main__":
    download_pdfs()