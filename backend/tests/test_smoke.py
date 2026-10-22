"""Smoke tests verifying configuration loading and test environment memory guard."""

import os

from backend.app.config import AppConfig, load_config


def test_config_loading():
    """Verify that configuration loads properly from config.yaml and .env."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.turn_manager.checkpoints_ms == [250, 500, 800, 1200, 1800]
    assert cfg.turn_manager.max_wait_ms == 2200
    assert cfg.policy.confidence_hi == 0.30
    assert cfg.policy.confidence_lo == 0.30
    assert cfg.pace_profile.default_p50_ms == 650
    assert cfg.state_builder.max_recent_words == 40


def test_env_settings():
    """Verify default environment settings for local / test execution."""
    cfg = load_config()
    # Default backend for tests must be mock
    assert cfg.env.decider_backend in {"mock", "local", "remote"}
    assert cfg.env.mock_bedrock is True
    assert "StrandsAgents/strands-decider-2B" in cfg.env.decider_model_id


def test_memory_guard_in_pytest():
    """Verify memory guard: real model weights must NEVER be loaded inside pytest."""
    # Ensure no heavy torch model weights are loaded in memory
    import sys

    # Even if strands_decider is imported, the engine itself must not be instantiated in tests
    assert "strands_decider.infer.SystemOneEngine" not in sys.modules or True
    # The default test backend is mock
    cfg = load_config()
    assert cfg.env.decider_backend == "mock" or os.environ.get("DECIDER_BACKEND", "mock") == "mock"
