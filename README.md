# Financial Agent RAG

> Assistant d'analyse financière augmenté par RAG (Retrieval-Augmented Generation) : un pipeline Big Data conteneurisé, de la collecte brute à l'agent conversationnel.

![Status](https://img.shields.io/badge/status-MVP%20fonctionnel-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-28%2B-blue)
![Kubernetes](https://img.shields.io/badge/kubernetes-1.30%2B-blue)
![Spark](https://img.shields.io/badge/spark-3.5.3-orange)
![Milvus](https://img.shields.io/badge/milvus-2.3.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Stack technique](#3-stack-technique)
4. [Structure du projet](#4-structure-du-projet)
5. [Démarrage rapide](#5-démarrage-rapide)
6. [Installation détaillée](#6-installation-détaillée)
7. [Pipeline de bout en bout](#7-pipeline-de-bout-en-bout)
8. [Utilisation de l'agent RAG](#8-utilisation-de-lagent-rag)
9. [Déploiement Kubernetes](#9-déploiement-kubernetes)
10. [Tests et validation](#10-tests-et-validation)
11. [Limitations connues](#11-limitations-connues)
12. [Roadmap](#12-roadmap)
13. [Contribuer et licence](#13-contribuer-et-licence)

---

## 1. Vue d'ensemble

**Financial Agent RAG** répond à des questions complexes sur des documents d'entreprises (10-K, 10-Q, présentations investisseurs, actualités, transcriptions d'earnings calls) en :

- extrayant l'information de sources **hétérogènes** (PDF, DOCX, HTML SEC, JSON) ;
- croisant plusieurs documents pour **comparer des entreprises ou des périodes** ;
- **citant les sources exactes** (`chunk_id`, page, entreprise, période) ;
- justifiant sa réponse par un **graphe de raisonnement**.

Chaque couche est conteneurisée, distribuée et conçue pour être déployée sur Kubernetes, puis sur AWS (EKS).

### État d'avancement (MVP)

| Couche | Détail | État |
|---|---|---|
| Collecte | SEC, FinQA, News, Transcripts, scrapers | ✅ Fonctionnel |
| Data Lake | MinIO (S3-compatible), 88 fichiers sources | ✅ Fonctionnel |
| Parsing distribué | Spark + 4 parsers, 7 149 documents JSON | ✅ Fonctionnel |
| Chunking distribué | Spark, 25 188 chunks JSONL | ✅ Fonctionnel |
| Embedding GPU | BGE-large, 25 188 vecteurs | ✅ Fonctionnel |
| Indexation vectorielle | Milvus (HNSW), 25 188 entités | ✅ Fonctionnel |
| Indexation lexicale | Elasticsearch (BM25), 25 188 documents | ✅ Fonctionnel |
| API + agent | FastAPI + LangGraph | ✅ Fonctionnel |
| Interface | Streamlit | ✅ Fonctionnel |
| Manifests Kubernetes | base + overlays | 🟡 Partiel (dev local) |
| Déploiement AWS EKS | Terraform, node groups GPU | ⏳ À venir |

---

## 2. Architecture

### Vue des composants

```text
┌───────────────────────────────────────────────────────────────┐
│ HÔTE (Windows / Linux / macOS)                                │
│  • Driver NVIDIA (≥ 535 pour CUDA 12.4)                       │
│  • Ollama + Qwen 2.5 7B (port 11434)                          │
└───────────────────────────┬───────────────────────────────────┘
                            │ host.docker.internal:11434
                            ▼
┌───────────────────────────────────────────────────────────────┐
│ DOCKER COMPOSE / KUBERNETES                                   │
│                                                               │
│  Traitement distribué (CPU)                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │ Spark Master │───│ Spark Worker1│   │ Spark Worker2│       │
│  └──────┬───────┘   └──────────────┘   └──────────────┘       │
│         │ parsing + chunking                                  │
│         ▼                                                     │
│  Stockage et index                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ MinIO        │ │ Milvus       │ │ Elasticsearch│           │
│  │ (Data Lake)  │ │ (vecteurs)   │ │ (BM25)       │           │
│  └──────▲───────┘ └──────▲───────┘ └──────▲───────┘           │
│         │                └────────┬───────┘                   │
│  ┌──────┴───────┐          ┌──────┴───────┐  ┌─────────────┐  │
│  │ Collector    │          │ API FastAPI  │◄─│ UI Streamlit│  │
│  │ (one-shot)   │          │ (GPU)        │  └─────────────┘  │
│  └──────────────┘          └──────────────┘                   │
└───────────────────────────────────────────────────────────────┘
```

### Flux de données

```text
Sources ──► raw-documents/ ──► parsed-documents/ ──► processed-chunks/
        1. Collect         2. Parse             3. Chunk
                                                      │
                                                4. Embed (GPU)
                                                      ▼
                                                 embeddings/
                                                      │
                                                5. Index
                                          ┌───────────┴───────────┐
                                        Milvus                Elasticsearch
                                          └───────────┬───────────┘
                                                      ▼
                                              6. Agent RAG
```

---

## 3. Stack technique

### Données et ML

| Composant | Version | Rôle |
|---|---|---|
| Apache Spark | 3.5.3 | Parsing et chunking distribués (CPU) |
| MinIO | latest | Data Lake compatible S3 |
| Milvus | 2.3.3 | Base vectorielle (HNSW, COSINE) |
| Elasticsearch | 8.11.0 | Recherche lexicale BM25 |
| Kibana | 8.11.0 | Visualisation Elasticsearch |
| etcd | 3.5.5 | Metadata store de Milvus |

### Modèles

| Modèle | Rôle | Taille |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | Embeddings (1024 dim) | ~1,3 Go |
| `BAAI/bge-reranker-base` | Reranking (cross-encoder) | ~1,1 Go |
| Qwen 2.5 7B (Ollama) | LLM de l'agent (via l'hôte) | ~4,7 Go |

### Applications

| Composant | Version | Rôle |
|---|---|---|
| FastAPI | ≥ 0.104 | API de l'agent RAG |
| Streamlit | ≥ 1.29 | Interface utilisateur |
| LangChain / LangGraph | latest | Orchestration de l'agent |
| pydantic-settings | ≥ 2.0 | Configuration centralisée |
| boto3 | ≥ 1.34 | Client S3 / MinIO |

### Infrastructure

| Composant | Version | Rôle |
|---|---|---|
| Docker Desktop | ≥ 28 | Runtime de conteneurs |
| Kubernetes | ≥ 1.30 | Orchestration (Docker Desktop, Kind, Minikube ou EKS) |
| NGINX Ingress | 1.10.0 | Exposition HTTP |
| Kustomize | intégré à kubectl | Overlays dev / prod |
| uv | latest | Gestionnaire de paquets Python |

---

## 4. Structure du projet

```text
financial-agent-rag/
├── docker/                      # Dockerfiles par service
│   ├── Dockerfile.base          #   Python 3.13 + CUDA + torch
│   ├── Dockerfile.api           #   FastAPI + embedding GPU
│   ├── Dockerfile.ui            #   Streamlit (CPU)
│   ├── Dockerfile.spark         #   Spark 3.5 + parsers
│   └── Dockerfile.collector     #   Collectors + scrapers + Playwright
│
├── infra/k8s/
│   ├── base/                    # Manifests communs
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── api.yaml
│   │   ├── ui.yaml
│   │   ├── ollama-external.yaml
│   │   ├── ingress.yaml
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── dev/                 # CPU, 1 réplica
│       └── prod/                # GPU, 2 réplicas
│
├── src/
│   ├── api/                     # FastAPI (main.py, routes/ask.py)
│   ├── config/settings.py       # Configuration centrale (variables d'env)
│   ├── core/
│   │   ├── storage.py           #   Abstraction S3 / local
│   │   ├── bm25.py              #   Indexeur Elasticsearch
│   │   ├── init_minio.py        #   Création des buckets
│   │   └── reset_databases.py   #   Reset Milvus + Elasticsearch
│   ├── ingestion/
│   │   ├── collectors/          #   collect_sec, collect_finqa, collect_news,
│   │   │                        #   collect_transcripts, collect_press_releases
│   │   ├── scrapers/            #   tesla, microsoft, alphabet (Playwright / curl-cffi)
│   │   ├── parsers/             #   base, pdf, docx, sec, json, manifest
│   │   └── chunking/
│   │       └── financial_chunker.py   # Chunking sensible aux tokens
│   ├── indexing/
│   │   ├── embedder.py          #   Génération des embeddings
│   │   └── milvus_indexer.py    #   Insertion dans Milvus
│   ├── rag/
│   │   ├── agent.py             #   Agent LangGraph
│   │   ├── search_engine.py     #   Recherche hybride
│   │   └── query_analyzer.py    #   Détection et décomposition des requêtes
│   └── ui/streamlit_app.py
│
├── spark_jobs/                  # Jobs Spark distribués
│   ├── parse_distributed.py
│   └── chunk_distributed.py
│
├── data/minio/                  # Volumes MinIO persistants (data/ legacy, migré vers MinIO)
├── docker-compose.yml
├── pyproject.toml               # Dépendances (uv)
├── uv.lock                      # Lockfile multi-plateforme
├── .env.example                 # Template de configuration
└── README.md
```

---

## 5. Démarrage rapide

Pour ceux qui veulent juste lancer le projet (détails dans les sections suivantes) :

```bash
git clone https://github.com/<your-org>/financial-agent-rag.git
cd financial-agent-rag
cp .env.example .env                       # puis adapter les valeurs

ollama serve &                             # sur l'hôte
ollama pull qwen2.5:7b

docker build -f docker/Dockerfile.base -t financial-agent-rag-base .
docker compose build api ui spark-master spark-worker
docker compose up -d minio etcd minio-milvus milvus elasticsearch kibana
docker compose exec api python -m src.core.init_minio

# Puis exécuter le pipeline (section 7) et lancer l'agent :
docker compose up -d api ui                # UI : http://localhost:8501
```

---

## 6. Installation détaillée

### 6.1 Prérequis

**Matériel**

| Composant | Minimum | Recommandé |
|---|---|---|
| OS | Windows 10 / macOS 12 / Linux | Windows 11 + WSL2 |
| RAM | 16 Go | 32 Go |
| CPU | 4 cœurs | 8 cœurs |
| Disque | 50 Go libres | 100 Go SSD |
| GPU | (optionnel) NVIDIA 4 Go | NVIDIA ≥ 6 Go (GTX 1660+, RTX) |

**Logiciels**

- Indispensables : Docker Desktop ≥ 28 (Kubernetes activé pour la partie K8s), Python 3.13, [uv](https://docs.astral.sh/uv/), kubectl ≥ 1.30, [Ollama](https://ollama.com).
- Optionnels : MSYS2 / Git Bash (Windows, pour `make` et le PATH Unix), Helm 3 (charts Milvus / ES / MinIO en K8s).

**Configuration GPU (NVIDIA uniquement)**

Sous Windows, le support est inclus dans Docker Desktop + WSL2. Sous Linux :

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Vérification :

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 6.2 Cloner le projet

```bash
git clone https://github.com/<your-org>/financial-agent-rag.git
cd financial-agent-rag
```

### 6.3 Configurer l'environnement

```bash
cp .env.example .env
```

Exemple de `.env` pour un développement local sous Windows :

```dotenv
# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password123
MINIO_BUCKET=financial-documents

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=financial_chunks

# Elasticsearch
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=financial_chunks_bm25

# Ollama / LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b

# Embedding / Reranker
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
FORCE_DEVICE=cuda            # cuda ou cpu

# Stockage
STORAGE_BACKEND=local        # local ou minio
```

> ⚠️ Les identifiants MinIO ci-dessus sont des valeurs de **développement uniquement**. Ne les réutilisez jamais en production et ne commitez pas votre `.env`.

### 6.4 Installer les dépendances Python (mode local)

```bash
uv sync
uv sync --extra ingestion --extra collectors --extra dev
```

### 6.5 Télécharger le LLM (sur l'hôte, dans un terminal séparé)

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 6.6 Construire les images Docker

```bash
# Image de base (une seule fois, ~10-15 min)
docker build -f docker/Dockerfile.base -t financial-agent-rag-base .

# Services applicatifs
docker compose build api ui spark-master spark-worker

# Collectors (one-shot)
docker compose --profile tools build collector
```

### 6.7 Démarrer l'infrastructure

```bash
# Milvus, Elasticsearch, MinIO, etc.
docker compose up -d minio etcd minio-milvus milvus elasticsearch kibana

# Attendre que Milvus et ES soient "healthy"
docker compose ps

# Créer les buckets MinIO
docker compose exec api python -m src.core.init_minio
```

---

## 7. Pipeline de bout en bout

> 💡 Sous Git Bash (Windows), préfixez les commandes `docker compose exec` par `MSYS_NO_PATHCONV=1` pour éviter la conversion des chemins.

### Étape 1 : Collecte des données brutes

```bash
# FinQA (dataset benchmark)
docker compose --profile tools run --rm collector python -m src.ingestion.collectors.collect_finqa

# SEC 10-K (AAPL, TSLA, GOOGL, MSFT)
docker compose --profile tools run --rm collector python -m src.ingestion.collectors.collect_sec

# News (Yahoo Finance)
docker compose --profile tools run --rm collector python -m src.ingestion.collectors.collect_news

# Transcripts d'earnings calls
docker compose --profile tools run --rm collector python -m src.ingestion.collectors.collect_transcripts

# 8-K (communiqués)
docker compose --profile tools run --rm collector python -m src.ingestion.collectors.collect_press_releases

# Scrapers (Playwright / curl-cffi)
docker compose --profile tools run --rm collector python -m src.ingestion.scrapers.tesla
docker compose --profile tools run --rm collector python -m src.ingestion.scrapers.microsoft
docker compose --profile tools run --rm collector python -m src.ingestion.scrapers.alphabet
```

**Résultat :** ~88 fichiers dans `raw-documents/` (98 Mo).

### Étape 2 : Parsing distribué (Spark)

Les paramètres S3A sont identiques pour les deux jobs Spark ; on les factorise dans une variable :

```bash
SPARK_S3_CONF="
  --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000
  --conf spark.hadoop.fs.s3a.access.key=admin
  --conf spark.hadoop.fs.s3a.secret.key=password123
  --conf spark.hadoop.fs.s3a.path.style.access=true
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem
"

docker compose exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  $SPARK_S3_CONF \
  --executor-memory 2G \
  --total-executor-cores 4 \
  /app/spark_jobs/parse_distributed.py \
  --prefix raw-documents/ \
  --partitions 4
```

**Résultat :** 7 149 documents JSON dans `parsed-documents/` (≈ 2-3 min sur 2 workers).

### Étape 3 : Chunking distribué (Spark)

```bash
docker compose exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  $SPARK_S3_CONF \
  --executor-memory 2G \
  --total-executor-cores 4 \
  /app/spark_jobs/chunk_distributed.py \
  --prefix parsed-documents/ \
  --partitions 8
```

**Résultat :** 25 188 chunks JSONL dans `processed-chunks/` (≈ 2 min).

### Étape 4 : Embedding (GPU)

```bash
# Test rapide (5 fichiers)
docker compose exec api python -m src.indexing.embedder --limit 5

# Run complet (~20-30 min en GPU, ~2-3 h en CPU)
docker compose exec api python -m src.indexing.embedder
```

**Résultat :** 25 188 chunks avec vecteurs 1024-dim dans `embeddings/`.

### Étape 5 : Indexation

```bash
# Milvus (vecteurs)
docker compose exec api python -m src.indexing.milvus_indexer

# Elasticsearch (BM25)
docker compose exec api python -m src.core.bm25
```

**Vérification** (attendu : `25188` dans les deux) :

```bash
# Milvus
docker compose exec api python -c "
from pymilvus import connections, Collection
from src.config.settings import settings
connections.connect(host=settings.milvus_host, port=str(settings.milvus_port))
c = Collection('financial_chunks')
c.load()
print(f'Milvus : {c.num_entities} entities')
"

# Elasticsearch
docker compose exec api python -c "
from elasticsearch import Elasticsearch
from src.config.settings import settings
es = Elasticsearch(settings.elasticsearch_host)
print(f'ES : {es.count(index=\"financial_chunks_bm25\")[\"count\"]} docs')
"
```

---

## 8. Utilisation de l'agent RAG

### 8.1 Démarrer les services applicatifs

```bash
docker compose up -d api ui
docker compose logs -f api      # attendre "Application startup complete"
```

### 8.2 API FastAPI

Documentation interactive : <http://localhost:8000/docs>

```bash
# Health check
curl http://localhost:8000/health

# Question simple
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Tesla Q4 2025 revenue?", "top_k": 5}'

# Question comparative
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare Alphabet and Microsoft revenue in 2024", "top_k": 5}'

# Question avec filtres
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risk factors?", "top_k": 3, "company": "TSLA", "document_type": "10-K"}'
```

### 8.3 Interface Streamlit

Ouvrir <http://localhost:8501>. Fonctionnalités :

- zone de chat ;
- réponse accompagnée de sources cliquables ;
- graphe de raisonnement (étapes de l'agent) ;
- filtres par entreprise, période et type de document.

---

## 9. Déploiement Kubernetes

### 9.1 Activer Kubernetes sur Docker Desktop

`Docker Desktop → Settings → Kubernetes → Enable`, attendre ~2 min, puis vérifier :

```bash
kubectl get nodes
```

### 9.2 Installer NGINX Ingress

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/cloud/deploy.yaml

kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

### 9.3 Déployer (dev local)

```bash
kubectl apply -k infra/k8s/overlays/dev
kubectl -n financial-rag get pods,svc,ingress
```

### 9.4 Ajouter l'hôte local

Éditer `C:\Windows\System32\drivers\etc\hosts` (en administrateur) :

```text
127.0.0.1 rag.local
```

### 9.5 Accéder aux services

- UI : <http://rag.local>
- API : <http://rag.local/api/ask>

Alternative par port-forward :

```bash
kubectl -n financial-rag port-forward svc/rag-api 8000:8000
kubectl -n financial-rag port-forward svc/rag-ui 8501:8501
```

### 9.6 (Futur) Infrastructure en K8s via Helm

```bash
# Milvus
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm install milvus milvus/milvus -n financial-rag --set standalone.enabled=true

# Elasticsearch
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install elasticsearch bitnami/elasticsearch -n financial-rag

# MinIO
helm install minio bitnami/minio -n financial-rag \
  --set auth.rootUser=admin --set auth.rootPassword=password123
```

---

## 10. Tests et validation

### Recherche hybride (test rapide)

```bash
docker compose exec api python -c "
from src.rag.search_engine import SearchEngine
engine = SearchEngine()
docs = engine.search_hybrid('Tesla Q4 2025 revenue', top_k=5)
for d in docs:
    print(f\"{d['company']} | {d['period']} | {d['document_type']} | score={d.get('rerank_score', 0):.3f}\")
"
```

### Benchmark séquentiel vs Spark

| Jeu de données | Séquentiel | Spark (2 workers) |
|---|---|---|
| 88 fichiers PDF / DOCX / SEC | ~10-15 min | ~4-5 min |
| 7 149 documents FinQA | ~5 min | ~2-3 min |

### Validation des stores

```bash
docker compose exec api python -c "
from src.core.storage import storage
for prefix in ['raw-documents/', 'parsed-documents/', 'processed-chunks/', 'embeddings/']:
    print(f'{prefix:<20}: {len(storage.list(prefix))} fichiers')
"
```

---

## 11. Limitations connues

### 11.1 Synthèse comparative multi-entreprises

Le pipeline retrouve correctement les chunks pertinents pour chaque entreprise, mais le LLM local Qwen 2.5 7B (quantization Q4) peut confondre les chiffres entre deux entreprises dans une question comparative.

| Type de question | Exemple | Précision |
|---|---|---|
| Factuelle simple | « Tesla Q4 2025 revenue » | ✅ ~100 % |
| Comparative | « Compare Alphabet vs Microsoft » | ⚠️ ~50 % |

Solutions prévues :

- Bedrock Claude 3.5 Sonnet (AWS) pour un cross-referencing fiable ;
- ou Qwen 2.5 14B / 72B via vLLM sur EC2 G5 ;
- ou une méthode structurée : un appel par entreprise + table de synthèse forcée.

### 11.2 Dépendance à Python 3.13

Le projet nécessite Python 3.13 car `torch 2.6+cu124` n'existe pas pour `cp313` sur Windows/Linux pour le moment. Le lockfile est multi-plateforme (Windows + Linux x86_64).

### 11.3 GPU et Docker Desktop Kubernetes

Docker Desktop K8s ne supporte pas nativement les GPU : en dev K8s, utiliser `FORCE_DEVICE=cpu`. En production EKS, des node groups GPU avec `nvidia-device-plugin` seront utilisés.

### 11.4 Dépréciations PyMilvus

`connections.connect()` et `Collection` sont dépréciés en PyMilvus 3.1. Migration vers `MilvusClient` prévue.

---

## 12. Roadmap

**✅ Fait**

- [x] Conteneurisation complète (Docker Compose, 8 services)
- [x] Data Lake MinIO (S3-compatible)
- [x] Parsing distribué (Spark + 4 parsers)
- [x] Chunking distribué (Spark)
- [x] Embedding GPU (BGE-large)
- [x] Indexation Milvus + Elasticsearch
- [x] Agent RAG (LangGraph + Ollama)
- [x] API FastAPI + UI Streamlit
- [x] GPU passthrough dans le conteneur API
- [x] Manifests Kubernetes (base + overlays)

**🟡 En cours**

- [ ] Déploiement K8s complet sur Docker Desktop (pods Running)
- [ ] Infrastructure K8s via Helm (Milvus, ES, MinIO)
- [ ] Ollama containerisé dans K8s

**⏳ À venir**

- [ ] **AWS EKS** : Terraform (VPC, EKS, node groups CPU + GPU), migration MinIO → S3, ES → OpenSearch Service, Bedrock (Claude 3.5) pour le LLM
- [ ] **CI/CD** : GitHub Actions (build + push GHCR/ECR), tests automatisés, déploiement via ArgoCD/Flux
- [ ] **Monitoring** : Prometheus + Grafana, logs centralisés (Loki ou CloudWatch)
- [ ] **Optimisations** : PyMilvus ORM → MilvusClient, embedding par batch plus gros (`batch_size=128`), cache Redis pour les questions fréquentes
- [ ] **Fonctionnalités** : extraction de tableaux vers Parquet structuré, fine-tuning des embeddings sur FinQA, multi-agents (un par entreprise), interface React avec surlignage PDF

---

## 13. Contribuer et licence

### Conventions de commit

```text
feat:      nouvelle fonctionnalité
fix:       correction de bug
docs:      documentation
chore:     maintenance
refactor:  refactorisation
test:      ajout de tests
```

### Licence

MIT, voir [LICENSE](LICENSE).

### Auteurs

Projet développé dans le cadre du cours **Big Data & Cloud** par une équipe de 3 étudiants (Ingestion / Agent RAG / Cloud-MLOps).

### Ressources

- [Apache Spark](https://spark.apache.org/docs/latest/)
- [Milvus](https://milvus.io/docs)
- [Elasticsearch](https://www.elastic.co/guide/index.html)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Kubernetes](https://kubernetes.io/docs/)
- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)

---

<p align="center"><b>⭐ Si ce projet vous aide, laissez une étoile !</b></p>