# -*- coding: utf-8 -*-
"""설정 한 곳. 값은 .env 에서 읽고, 코드에는 기본값만 둔다."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# HuggingFace tokenizers 는 기본으로 스레드를 띄운다. 임베딩 모델을 쓴 뒤 리랭커를
# 로드하면 그 스레드 때문에 교착에 빠져 CPU 0% 로 멈춘다(실제로 겪었다).
# 임포트 시점에 꺼 둔다. 성능 손해는 배치 처리에서 거의 없다.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── LLM ──
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model_main: str = "openai:gpt-5.4-mini"
    llm_model_small: str = "openai:gpt-5.4-nano"
    llm_model_local: str = "ollama:qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"

    # ── 임베딩·리랭커 ──
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 8
    # RRF 로 모을 후보 수. 리랭커 입력이 되므로 지연과 메모리를 직접 좌우한다.
    # 16GB 노트북에서 BGE-M3(2.2GB)와 리랭커(2.2GB)를 함께 올리면 여유가 없어서
    # 후보를 50개로 두면 스왑이 터진다(실측). 30개가 타협점이다.
    rerank_candidates: int = 10
    hf_home: str = "./models"

    # ── Qdrant ──
    # 비어 있으면 임베디드 모드(qdrant_path)로 뜬다. 도커 없이도 수업이 굴러가야 한다.
    qdrant_url: str = ""
    qdrant_path: str = "./data/qdrant_local"
    qdrant_collection: str = "finrag"

    # ── Langfuse ──
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── API ──
    api_key: str = "change-me"
    rate_limit: str = "30/minute"

    # ── 에이전트 정책 ──
    max_retries: int = 2

    # ── 경로 ──
    @property
    def root(self) -> Path:
        return ROOT

    @property
    def data(self) -> Path:
        return ROOT / "data"

    @property
    def chunks_dir(self) -> Path:
        return ROOT / "data" / "chunks"

    @property
    def results_dir(self) -> Path:
        return ROOT / "results"

    @property
    def golden_dir(self) -> Path:
        return ROOT / "data" / "golden"

    @property
    def has_llm_key(self) -> bool:
        return bool(self.openai_api_key or self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
