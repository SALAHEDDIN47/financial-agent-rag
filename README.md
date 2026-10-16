# Financial Agent RAG

> Assistant d'analyse financière augmenté par RAG (Retrieval-Augmented
> Generation) --- Pipeline Big Data conteneurisé, de la collecte brute à
> l'agent conversationnel.

![Status](https://img.shields.io/badge/status-MVP%20fonctionnel-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-28%2B-blue)
![Kubernetes](https://img.shields.io/badge/kubernetes-1.30%2B-blue)
![Spark](https://img.shields.io/badge/spark-3.5.3-orange)
![Milvus](https://img.shields.io/badge/milvus-2.3.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)

------------------------------------------------------------------------

## 📖 Table des matières

1.  [Vue d'ensemble](#1-vue-densemble)
2.  [Architecture](#2-architecture)
3.  [Stack technique](#3-stack-technique)
4.  [Structure du projet](#4-structure-du-projet)
5.  [Prérequis](#5-prérequis)
6.  [Installation](#6-installation)
7.  [Pipeline de bout en bout](#7-pipeline-de-bout-en-bout)
8.  [Utilisation de l'agent RAG](#8-utilisation-de-lagent-rag)
9.  [Déploiement Kubernetes](#9-déploiement-kubernetes)
10. [Tests & validation](#10-tests--validation)
11. [Limitations connues](#11-limitations-connues)
12. [Roadmap](#12-roadmap)
13. [Contributions & licence](#13-contributions--licence)

------------------------------------------------------------------------

## 1. Vue d'ensemble

**Financial Agent RAG** est un système d'analyse financière capable de
répondre à des questions complexes sur des rapports d'entreprises (10-K,
10-Q, présentations investisseurs, actualités, transcriptions d'earnings
calls) en :

-   extrayant l'information de sources **hétérogènes** (PDF, DOCX, HTML
    SEC, JSON),
-   croisant plusieurs documents pour **comparer des entreprises** ou
    des périodes,
-   **citant les sources exactes** (chunk_id, page, entreprise,
    période),
-   justifiant sa réponse avec un **graphe de raisonnement**.

Le projet suit une architecture **Big Data / MLOps** : chaque couche est
conteneurisée, distribuée et prête à être déployée sur Kubernetes puis
sur AWS (EKS).

### Ce qui fonctionne à ce stade (MVP)

  Couche                                               État
  ---------------------------------------------------- -------------------------
  Collecte (SEC, FinQA, News, Transcripts, scrapers)   ✅ Fonctionnel
  Data Lake MinIO (S3-compatible)                      ✅ 88 fichiers sources
  Parsing distribué (Spark + 4 parsers)                ✅ 7 149 documents JSON
  Chunking distribué (Spark)                           ✅ 25 188 chunks JSONL
  Embedding GPU (BGE-large)                            ✅ 25 188 vecteurs
  Indexation Milvus (HNSW)                             ✅ 25 188 entités
  Indexation Elasticsearch (BM25)                      ✅ 25 188 documents
  API FastAPI + agent LangGraph                        ✅ Fonctionnel
  UI Streamlit                                         ✅ Fonctionnel
  Manifests Kubernetes (base + overlays)               🟡 Partiel (dev local)
  Déploiement AWS EKS                                  ⏳ À venir

------------------------------------------------------------------------

## 2. Architecture

┌──────────────────────────────────────────────────────────────────────┐
│ HÔTE (Windows / Linux / Mac) │ │ • NVIDIA Driver (≥ 535 pour CUDA
12.4) │ │ • Ollama + Qwen 2.5 7B (port 11434) --- appelé par les
conteneurs │
└────────────────────────────┬─────────────────────────────────────────┘
│ host.docker.internal:11434 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ DOCKER COMPOSE / KUBERNETES │ │ │ │ ┌─────────────┐ ┌─────────────┐
┌─────────────┐ │ │ │ SPARK │──│ SPARK │──│ SPARK │ CPU │ │ │ MASTER │ │
WORKER 1 │ │ WORKER 2 │ │ │ └──────┬──────┘ └─────────────┘
└─────────────┘ │ │ │ Parsing + Chunking distribués │ │ ▼ │ │
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │ │ MinIO │ │ Milvus │
│ Elastic- │ │ │ │ Data Lake │ │ Vecteurs │ │ search BM25 │ │ │
└─────────────┘ └─────────────┘ └─────────────┘ │ │ ▲ ▲ ▲ │ │ │ │ │ │ │
┌──────┴──────┐ ┌────┴─────┐ ┌────────┴────┐ │ │ │ API GPU │ │ UI │ │
COLLECTOR │ │ │ │ FastAPI │◄─│Streamlit │ │ one-shot │ │ │
└─────────────┘ └──────────┘ └─────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

``` text
```

### Flux de données

Sources raw-documents/ parsed-documents/ processed-chunks/ │ │ │ │ │ 1.
Collect │ 2. Parse │ 3. Chunk │
├───────────────────►├───────────────────►├───────────────────►│ │ │ │ │
│ │ │ 4. Embed │ │ │ │ ┌─────────────────┤ │ │ │ ▼ │ │ │ │ embeddings/ │
│ │ │ │ │ │ │ │ │ 5. Index │ │ │ │ ▼ │ │ │ Milvus ─┐ │ │ │ ES ─────┴───►
6. RAG Agent │

``` text

---
```

## 3. Stack technique

### Data & ML

  Composant       Version   Rôle
  --------------- --------- -------------------------------------
  Apache Spark    3.5.3     Parsing + chunking distribués (CPU)
  MinIO           latest    Data Lake S3-compatible
  Milvus          2.3.3     Base vectorielle (HNSW, COSINE)
  Elasticsearch   8.11.0    Recherche lexicale BM25
  Kibana          8.11.0    Visualisation ES
  etcd            3.5.5     Metadata store Milvus

### Modèles

  Modèle                   Rôle                       Taille
  ------------------------ -------------------------- ----------
  BAAI/bge-large-en-v1.5   Embeddings (1024 dim)      \~1.3 GB
  BAAI/bge-reranker-base   Cross-encoder reranking    \~1.1 GB
  Qwen 2.5 7B (Ollama)     LLM agent (via API hôte)   \~4.7 GB

### Applications

  Composant               Version   Rôle
  ----------------------- --------- ---------------------------
  FastAPI                 ≥0.104    API agent RAG
  Streamlit               ≥1.29     Interface utilisateur
  LangChain / LangGraph   latest    Orchestration de l'agent
  Pydantic-settings       ≥2.0      Configuration centralisée
  boto3                   ≥1.34     Client S3/MinIO

### Infrastructure

  -----------------------------------------------------------------------
  Composant               Version                 Rôle
  ----------------------- ----------------------- -----------------------
  Docker Desktop          ≥28                     Runtime conteneurs

  Kubernetes              ≥1.30                   Orchestration (via
                                                  Docker Desktop, Kind,
                                                  Minikube ou EKS)

  NGINX Ingress           1.10.0                  Exposition HTTP

  Kustomize               intégré kubectl         Gestion des overlays
                                                  (dev/prod)

  uv                      latest                  Gestionnaire Python
                                                  ultra-rapide
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. Structure du projet

financial-agent-rag/ ├── docker/ \# Dockerfiles par service │ ├──
Dockerfile.base \# Python 3.13 + CUDA + torch │ ├── Dockerfile.api \#
FastAPI + embedding GPU │ ├── Dockerfile.ui \# Streamlit (CPU) │ ├──
Dockerfile.spark \# Spark 3.5 + parsers │ └── Dockerfile.collector \#
Collectors + scrapers + Playwright │ ├── infra/ │ └── k8s/ │ ├── base/
\# Manifests Kubernetes communs │ │ ├── namespace.yaml │ │ ├──
configmap.yaml │ │ ├── secrets.yaml │ │ ├── api.yaml │ │ ├── ui.yaml │ │
├── ollama-external.yaml │ │ ├── ingress.yaml │ │ └── kustomization.yaml
│ └── overlays/ │ ├── dev/ \# Overrides dev (CPU, 1 replica) │ │ └──
kustomization.yaml │ └── prod/ \# Overrides prod (GPU, 2 replicas) │ └──
kustomization.yaml │ ├── src/ │ ├── api/ \# FastAPI │ │ ├── main.py │ │
└── routes/ │ │ └── ask.py │ │ │ ├── config/ │ │ └── settings.py \#
Config centrale (env-based) │ │ │ ├── core/ │ │ ├── storage.py \#
Abstraction S3 / local │ │ ├── bm25.py \# Indexer Elasticsearch │ │ ├──
init_minio.py \# Création des buckets │ │ └── reset_databases.py \#
Reset Milvus + ES │ │ │ ├── ingestion/ │ │ ├── collectors/ \#
Téléchargement de données brutes │ │ │ ├── collect_sec.py │ │ │ ├──
collect_finqa.py │ │ │ ├── collect_news.py │ │ │ ├──
collect_transcripts.py │ │ │ └── collect_press_releases.py │ │ │ │ │ ├──
scrapers/ \# Playwright / curl-cffi │ │ │ ├── tesla.py │ │ │ ├──
microsoft.py │ │ │ └── alphabet.py │ │ │ │ │ ├── parsers/ \# PDF / DOCX
/ SEC / JSON → JSON │ │ │ ├── base_parser.py │ │ │ ├── pdf_parser.py │ │
│ ├── docx_parser.py │ │ │ ├── sec_parser.py │ │ │ ├── json_parser.py │
│ │ └── manifest.py │ │ │ │ │ └── chunking/ │ │ └── financial_chunker.py
\# Chunking token-aware │ │ │ ├── indexing/ │ │ ├── embedder.py \#
Génère les embeddings │ │ └── milvus_indexer.py \# Insère dans Milvus │
│ │ ├── rag/ │ │ ├── agent.py \# Agent LangGraph │ │ ├──
search_engine.py \# Recherche hybride │ │ └── query_analyzer.py \#
Détection + décomposition │ │ │ └── ui/ │ └── streamlit_app.py │ ├──
spark_jobs/ \# Jobs Spark distribués │ ├── parse_distributed.py │ └──
chunk_distributed.py │ ├── data/ \# (legacy --- migré vers MinIO) │ └──
minio/ \# Volumes MinIO persistants │ ├── docker-compose.yml \#
Orchestration Docker ├── pyproject.toml \# Dépendances Python (uv) ├──
uv.lock \# Lockfile (multi-plateforme) ├── .env.example \# Template de
configuration └── README.md

``` text

---
```

## 5. Prérequis

### Système

  Composant    Minimum                         Recommandé
  ------------ ------------------------------- --------------------------------
  **OS**       Windows 10 / macOS 12 / Linux   Windows 11 + WSL2
  **RAM**      16 GB                           32 GB
  **CPU**      4 cores                         8 cores
  **Disque**   50 GB libres                    100 GB SSD
  **GPU**      (optionnel) NVIDIA 4 GB         NVIDIA ≥ 6 GB (GTX 1660+, RTX)

### Logiciels

``` bash
# Indispensables
- Docker Desktop ≥ 28 (avec Kubernetes activé pour K8s)
- Python 3.13
- uv (https://docs.astral.sh/uv/)
- kubectl ≥ 1.30
- Ollama (https://ollama.com)

# Optionnels mais recommandés
- MSYS2 / Git Bash (Windows uniquement, pour `make` et PATH Unix)
- Helm 3 (pour les charts Milvus/ES/MinIO en K8s)
Configuration GPU (NVIDIA uniquement)
```bash
```

# 1. Installer NVIDIA Container Toolkit

# Windows : inclus dans Docker Desktop + WSL2

# Linux :

distribution=\$(. /etc/os-release;echo $ID$VERSION_ID) curl -s -L
https://nvidia.github.io/libnvidia-container/gpgkey \| sudo apt-key
add - curl -s -L
https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list
\|\
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list sudo
apt-get update && sudo apt-get install -y nvidia-container-toolkit sudo
systemctl restart docker

# 2. Vérifier

docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04
nvidia-smi 6. Installation 6.1 Cloner le projet

``` bash
git clone https://github.com/<your-org>/financial-agent-rag.git
cd financial-agent-rag
6.2 Créer le fichier .env
cp .env.example .env
```

# Édite .env selon ton environnement

Contenu .env (dev local Windows) :

``` dotenv
```

# MinIO

MINIO_ENDPOINT=localhost:9000 MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password123 MINIO_BUCKET=financial-documents

# Milvus

MILVUS_HOST=localhost MILVUS_PORT=19530
MILVUS_COLLECTION=financial_chunks

# Elasticsearch

ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=financial_chunks_bm25

# Ollama / LLM

OLLAMA_BASE_URL=http://localhost:11434 LLM_MODEL=qwen2.5:7b

# Embedding / Reranker

EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base FORCE_DEVICE=cuda \# ou cpu

# Storage

STORAGE_BACKEND=local \# ou minio 6.3 Installer les dépendances Python
(local)

``` bash
uv sync
uv sync --extra ingestion --extra collectors --extra dev
6.4 Télécharger les modèles LLM
```

# Dans un terminal séparé (host)

ollama serve ollama pull qwen2.5:7b 6.5 Construire les images Docker

``` bash
```

# Image de base (une seule fois, \~10-15 min)

docker build -f docker/Dockerfile.base -t financial-agent-rag-base .

# Services applicatifs

docker compose build api ui spark-master spark-worker

# Collectors (one-shot)

docker compose --profile tools build collector 6.6 Démarrer
l'infrastructure

``` bash
```

# Infra d'abord (Milvus, ES, MinIO, etc.)

docker compose up -d minio etcd minio-milvus milvus elasticsearch kibana

# Attendre que Milvus + ES soient healthy

docker compose ps

# Créer les buckets MinIO

docker compose exec api python -m src.core.init_minio 7. Pipeline de
bout en bout 📥 Étape 1 --- Collecte des données brutes

``` bash
```

# FinQA (dataset benchmark)

docker compose --profile tools run --rm collector\
python -m src.ingestion.collectors.collect_finqa

# SEC 10-K (AAPL, TSLA, GOOGL, MSFT)

docker compose --profile tools run --rm collector\
python -m src.ingestion.collectors.collect_sec

# News (Yahoo Finance)

docker compose --profile tools run --rm collector\
python -m src.ingestion.collectors.collect_news

# Transcripts d'earnings calls

docker compose --profile tools run --rm collector\
python -m src.ingestion.collectors.collect_transcripts

# 8-K (communiqués)

docker compose --profile tools run --rm collector\
python -m src.ingestion.collectors.collect_press_releases

# Scrapers (Playwright / curl-cffi)

docker compose --profile tools run --rm collector\
python -m src.ingestion.scrapers.tesla docker compose --profile tools
run --rm collector\
python -m src.ingestion.scrapers.microsoft docker compose --profile
tools run --rm collector\
python -m src.ingestion.scrapers.alphabet Résultat : \~88 fichiers dans
raw-documents/ (98 MB).

📄 Étape 2 --- Parsing distribués (Spark)

``` bash
docker compose exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
    --conf spark.hadoop.fs.s3a.access.key=admin \
    --conf spark.hadoop.fs.s3a.secret.key=password123 \
    --conf spark.hadoop.fs.s3a.path.style.access=true \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --executor-memory 2G \
    --total-executor-cores 4 \
    /app/spark_jobs/parse_distributed.py \
    --prefix raw-documents/ \
    --partitions 4
```

Résultat : 7 149 documents JSON dans parsed-documents/ (≈ 2-3 min sur 2
workers).

✂️ Étape 3 --- Chunking distribué (Spark)

``` bash
docker compose exec spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
    --conf spark.hadoop.fs.s3a.access.key=admin \
    --conf spark.hadoop.fs.s3a.secret.key=password123 \
    --conf spark.hadoop.fs.s3a.path.style.access=true \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --executor-memory 2G \
    --total-executor-cores 4 \
    /app/spark_jobs/chunk_distributed.py \
    --prefix parsed-documents/ \
    --partitions 8
```

Résultat : 25 188 chunks JSONL dans processed-chunks/ (≈ 2 min).

🧠 Étape 4 --- Embedding GPU

``` bash
```

# Test rapide (5 fichiers)

MSYS_NO_PATHCONV=1 docker compose exec api python -m
src.indexing.embedder --limit 5

# Run complet (attention : \~2-3h en CPU, \~20-30 min en GPU)

MSYS_NO_PATHCONV=1 docker compose exec api python -m
src.indexing.embedder Résultat : 25 188 chunks avec vecteurs 1024-dim
dans embeddings/.

🔍 Étape 5 --- Indexation

``` bash
```

# Milvus (vecteurs)

MSYS_NO_PATHCONV=1 docker compose exec api python -m
src.indexing.milvus_indexer

# Elasticsearch (BM25)

MSYS_NO_PATHCONV=1 docker compose exec api python -m src.core.bm25
Vérification :

``` bash
```

# Milvus

MSYS_NO_PATHCONV=1 docker compose exec api python -c " from pymilvus
import connections, Collection from src.config.settings import settings
connections.connect(host=settings.milvus_host,
port=str(settings.milvus_port)) c = Collection('financial_chunks')
c.load() print(f'Milvus : {c.num_entities} entities') "

# Elasticsearch

docker compose exec api python -c " from elasticsearch import
Elasticsearch from src.config.settings import settings es =
Elasticsearch(settings.elasticsearch_host) print(f'ES :
{es.count(index=\"financial_chunks_bm25\")\[\"count\"\]} docs') "
Attendu : 25188 dans les deux.

8.  Utilisation de l'agent RAG 8.1 Démarrer les services applicatifs

``` bash
docker compose up -d api ui
docker compose logs -f api
Attendre Application startup complete.

8.2 API FastAPI — Test direct
```

# Health check

curl http://localhost:8000/health

# Question simple

curl -X POST http://localhost:8000/api/ask\
-H "Content-Type: application/json"\
-d '{"query": "What was Tesla Q4 2025 revenue?", "top_k": 5}'

# Question comparative

curl -X POST http://localhost:8000/api/ask\
-H "Content-Type: application/json"\
-d '{"query": "Compare Alphabet and Microsoft revenue in 2024", "top_k":
5}'

# Question avec filtres

curl -X POST http://localhost:8000/api/ask\
-H "Content-Type: application/json"\
-d '{"query": "What are the risk factors?", "top_k": 3, "company":
"TSLA", "document_type": "10-K"}' Documentation interactive :
http://localhost:8000/docs

8.3 UI Streamlit Navigateur → http://localhost:8501

Fonctionnalités :

Zone de chat

Affichage de la réponse + sources cliquables

Graphe de raisonnement (étapes de l'agent)

Filtres par entreprise / période / type

9.  Déploiement Kubernetes 9.1 Activer Kubernetes sur Docker Desktop
    Docker Desktop → Settings → Kubernetes → ✅ Enable

Attendre \~2 min

Vérifier : kubectl get nodes

9.2 Installer NGINX Ingress

``` bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/cloud/deploy.yaml

kubectl wait --namespace ingress-nginx \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/component=controller \
    --timeout=120s
9.3 Déployer (dev local)
kubectl apply -k infra/k8s/overlays/dev

kubectl -n financial-rag get pods,svc,ingress
9.4 Ajouter les hôtes locaux
Édite C:\Windows\System32\drivers\etc\hosts (admin) :

127.0.0.1 rag.local
9.5 Accéder
```

UI : http://rag.local

API : http://rag.local/api/ask

Port-forward alternative :

``` bash
kubectl -n financial-rag port-forward svc/rag-api 8000:8000
kubectl -n financial-rag port-forward svc/rag-ui 8501:8501
9.6 (Futur) Infra en K8s via Helm
```

# Milvus

helm repo add milvus https://zilliztech.github.io/milvus-helm/ helm
install milvus milvus/milvus -n financial-rag --set
standalone.enabled=true

# Elasticsearch

helm repo add bitnami https://charts.bitnami.com/bitnami helm install
elasticsearch bitnami/elasticsearch -n financial-rag

# MinIO

helm install minio bitnami/minio -n financial-rag\
--set auth.rootUser=admin --set auth.rootPassword=password123 10. Tests
& validation Test unitaire rapide (pipeline)

``` bash
```

# Recherche hybride

docker compose exec api python -c " from src.rag.search_engine import
SearchEngine engine = SearchEngine() docs = engine.search_hybrid('Tesla
Q4 2025 revenue', top_k=5) for d in docs: print(f\"{d\['company'\]} \|
{d\['period'\]} \| {d\['document_type'\]} \|
score={d.get('rerank_score', 0):.3f}\") " Benchmark séquentiel vs Spark
Dataset Séquentiel Spark (2 workers) 88 fichiers PDF/DOCX/SEC \~10-15
min \~4-5 min 7 149 docs FinQA \~5 min \~2-3 min Validation des stores

``` bash
docker compose exec api python -c "
from src.core.storage import storage
print(f'raw-documents/      : {len(storage.list(\"raw-documents/\"))} fichiers')
print(f'parsed-documents/   : {len(storage.list(\"parsed-documents/\"))} fichiers')
print(f'processed-chunks/   : {len(storage.list(\"processed-chunks/\"))} fichiers')
print(f'embeddings/         : {len(storage.list(\"embeddings/\"))} fichiers')
"
```

11. Limitations connues 11.1 Synthèse comparative multi-entreprises Le
    pipeline retourne correctement les chunks pertinents pour chaque
    entreprise, mais le LLM local Qwen 2.5 7B (quantization Q4) peut
    confondre les chiffres entre 2 entreprises dans une question
    comparative.

Impact :

Questions factuelles simples ("Tesla Q4 2025 revenue") → ✅ précision
\~100%

Questions comparatives ("Compare Alphabet vs Microsoft") → ⚠️ \~50%
(peut mélanger)

Solutions prévues :

Bedrock Claude 3.5 Sonnet (AWS) → cross-referencing fiable

Ou Qwen 2.5 14B / 72B via vLLM sur EC2 G5

Ou méthode structurée : appel par entreprise + table de synthèse forcée

11.2 Dépendances à Python 3.13 Le projet nécessite Python 3.13 car torch
2.6+cu124 n'existe pas pour cp313 sur Windows/Linux pour l'instant. Le
lockfile est multi-plateforme (Windows + Linux x86_64).

11.3 GPU passthrough Docker Desktop K8s Docker Desktop K8s ne supporte
pas nativement les GPU. En dev K8s, on utilise FORCE_DEVICE=cpu. En prod
EKS, on aura des node groups GPU avec nvidia-device-plugin.

11.4 Warnings de dépréciation PyMilvus connections.connect() et
Collection sont dépréciés en PyMilvus 3.1 → migration vers MilvusClient
prévue.

12. Roadmap ✅ Fait ☑ Conteneurisation complète (Docker Compose, 8
    services) ☑ Data Lake MinIO (S3-compatible) ☑ Parsing distribué
    (Spark + 4 parsers) ☑ Chunking distribué (Spark) ☑ Embedding GPU
    (BGE-large) ☑ Indexation Milvus + Elasticsearch ☑ Agent RAG
    (LangGraph + Ollama) ☑ API FastAPI + UI Streamlit ☑ GPU passthrough
    dans le conteneur API ☑ Manifests Kubernetes (base + overlays) 🟡 En
    cours □ Déploiement K8s complet sur Docker Desktop (pods Running) □
    Infra K8s via Helm charts (Milvus, ES, MinIO) □ Ollama containerisé
    dans K8s ⏳ À venir □ Déploiement AWS EKS Terraform (VPC, EKS, node
    groups CPU + GPU)

Migration MinIO → S3

Migration ES → OpenSearch Service

Bedrock (Claude 3.5) pour le LLM

□ CI/CD GitHub Actions Build + push images vers GHCR/ECR

Tests automatisés

Déploiement K8s via ArgoCD/Flux

□ Monitoring Prometheus + Grafana

Logs centralisés (Loki ou CloudWatch)

□ Optimisations Migration PyMilvus ORM → MilvusClient

Embedding par batch plus gros (batch_size=128)

Cache Redis pour questions fréquentes

□ Fonctionnalités Extraction de tableaux vers Parquet structuré

Fine-tuning embeddings sur FinQA

Multi-agents (un par entreprise)

Interface React + surlignage PDF

13. Contributions & licence Conventions Git

``` text
feat:     Nouvelle fonctionnalité
fix:      Correction de bug
docs:     Documentation
chore:    Maintenance
refactor: Refactorisation
test:     Ajout de tests
```

Licence MIT --- voir LICENSE.

Auteurs Projet développé dans le cadre du cours Big Data & Cloud ---
architecture 3 étudiants (Ingestion / Agent RAG / Cloud-MLOps).

📚 Ressources Apache Spark Docs

Milvus Docs

Elasticsearch Docs

LangGraph Docs

Kubernetes Docs

AWS EKS Best Practices

```{=html}
<p align="center">
```
`<b>`{=html}⭐ Si ce projet vous aide, laissez une étoile !`</b>`{=html}
```{=html}
</p>
```
\`\`\`
