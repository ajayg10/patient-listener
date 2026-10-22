"""Application configuration loading from .env and config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TurnManagerConfig(BaseModel):
    checkpoints_ms: list[int] = Field(default_factory=lambda: [250, 500, 800, 1200, 1800])
    max_wait_ms: int = 2200
    silence_debounce_ms: int = 50


class PolicyConfig(BaseModel):
    confidence_hi: float = 0.30
    confidence_lo: float = 0.30
    operating_point: str = "Point A (Delay budget <= 800ms)"
    operating_point_provenance: str = "DEV-selected Point A: hi=0.30, lo=0.30 (DEV delay P50 <= 800ms, minimizing premature rate)"
    reask_interval_ms: int = 300


class PaceProfileConfig(BaseModel):
    default_p50_ms: int = 650
    default_p90_ms: int = 1100
    rolling_window: int = 20
    min_observations: int = 3


class StateBuilderConfig(BaseModel):
    max_recent_words: int = 40
    include_trailing_markers: bool = True
    include_pace_summary: bool = True


class EnvSettings(BaseSettings):
    decider_backend: str = "mock"  # mock | local | remote
    decider_url: str = "http://localhost:8000"
    decider_model_id: str = "StrandsAgents/strands-decider-2B-hobson-v19"
    decider_device: str = "cpu"  # cpu | cuda (GPU hosts use cuda)
    decider_torch_threads: int = 6
    hf_offline: bool = False

    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    aws_region: str = "us-east-1"
    mock_bedrock: bool = True

    hf_home: str = Field(
        default_factory=lambda: os.environ.get(
            "HF_HOME",
            "E:\\hf-cache" if os.name == "nt" and Path("E:/hf-cache").exists() else str(Path.home() / ".cache" / "huggingface")
        )
    )
    regret_db_path: str = "backend/app/data/regrets.db"
    config_path: str = "config.yaml"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    fixed_replay_timeout_ms: int = 800
    replay_decision_latency_ms: float = 301.2
    decision_cache_path: str = "backend/eval/data/decisions_cache_real.jsonl"
    demo_scenarios_path: str = "backend/eval/data/demo_scenarios.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppConfig(BaseModel):
    env: EnvSettings = Field(default_factory=EnvSettings)
    turn_manager: TurnManagerConfig = Field(default_factory=TurnManagerConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    pace_profile: PaceProfileConfig = Field(default_factory=PaceProfileConfig)
    state_builder: StateBuilderConfig = Field(default_factory=StateBuilderConfig)

    def __init__(self, **data: Any) -> None:
        env_kwargs = {}
        for k in list(data.keys()):
            if k in EnvSettings.model_fields:
                env_kwargs[k] = data.pop(k)
        if env_kwargs:
            existing_env = data.get("env")
            if existing_env is None:
                data["env"] = EnvSettings(**env_kwargs)
            elif isinstance(existing_env, EnvSettings):
                for ek, ev in env_kwargs.items():
                    setattr(existing_env, ek, ev)
        super().__init__(**data)

    @property
    def decider_backend(self) -> str:
        return self.env.decider_backend

    @decider_backend.setter
    def decider_backend(self, val: str) -> None:
        self.env.decider_backend = val

    @property
    def decider_url(self) -> str:
        return self.env.decider_url

    @decider_url.setter
    def decider_url(self, val: str) -> None:
        self.env.decider_url = val

    @property
    def decider_model_id(self) -> str:
        return self.env.decider_model_id

    @decider_model_id.setter
    def decider_model_id(self, val: str) -> None:
        self.env.decider_model_id = val

    @property
    def decider_device(self) -> str:
        return self.env.decider_device

    @decider_device.setter
    def decider_device(self, val: str) -> None:
        self.env.decider_device = val

    @property
    def decider_torch_threads(self) -> int:
        return self.env.decider_torch_threads

    @decider_torch_threads.setter
    def decider_torch_threads(self, val: int) -> None:
        self.env.decider_torch_threads = val

    @property
    def hf_home(self) -> str:
        return self.env.hf_home

    @property
    def regret_db_path(self) -> str:
        return self.env.regret_db_path

    @regret_db_path.setter
    def regret_db_path(self, val: str) -> None:
        self.env.regret_db_path = val

    @property
    def cors_origins(self) -> list[str]:
        return self.env.cors_origins

    @property
    def fixed_replay_timeout_ms(self) -> int:
        return self.env.fixed_replay_timeout_ms

    @property
    def replay_decision_latency_ms(self) -> float:
        return self.env.replay_decision_latency_ms

    @property
    def decision_cache_path(self) -> str:
        return self.env.decision_cache_path

    @property
    def demo_scenarios_path(self) -> str:
        return self.env.demo_scenarios_path


def load_config(config_yaml_path: str | Path | None = None) -> AppConfig:
    """Load configuration from environment variables (.env) and config.yaml."""
    env = EnvSettings()
    yaml_path = Path(config_yaml_path or env.config_path)

    yaml_data: dict[str, Any] = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    return AppConfig(
        env=env,
        turn_manager=TurnManagerConfig(**yaml_data.get("turn_manager", {})),
        policy=PolicyConfig(**yaml_data.get("policy", {})),
        pace_profile=PaceProfileConfig(**yaml_data.get("pace_profile", {})),
        state_builder=StateBuilderConfig(**yaml_data.get("state_builder", {})),
    )


def get_config(config_yaml_path: str | Path | None = None) -> AppConfig:
    """Convenience alias for load_config."""
    return load_config(config_yaml_path)


# Default global settings singleton
settings = load_config()
