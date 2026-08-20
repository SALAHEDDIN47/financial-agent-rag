.PHONY: install up down clean

install:
	uv pip install -e .  # ou poetry install

up:
	docker-compose up -d
	@echo "✅ Services démarrés (MinIO:9000, ES:9200, Milvus:19530, Jupyter:8888)"

down:
	docker-compose down -v

collect-data:
	python -m src.ingestion.collector

logs:
	docker-compose logs -f