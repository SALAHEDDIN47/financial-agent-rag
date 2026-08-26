from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "admin"
    minio_secret_key: str = "password123"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    elasticsearch_host: str = "http://localhost:9200"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()