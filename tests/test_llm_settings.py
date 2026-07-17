from __future__ import annotations

import pytest

from core.services.llm_settings import generation_settings

# ── Seed handling ────────────────────────────────────────────────────


def test_blank_seed_absent() -> None:
    result = generation_settings({"temperature": 0.5})
    assert "seed" not in result


def test_explicit_seed_zero_retained() -> None:
    result = generation_settings({"seed": 0})
    assert result["seed"] == 0


def test_explicit_seed_retained() -> None:
    result = generation_settings({"seed": 42})
    assert result["seed"] == 42


def test_seed_none_absent() -> None:
    result = generation_settings({"seed": None, "temperature": 0.5})
    assert "seed" not in result


# ── Range validation ──────────────────────────────────────────────────


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        generation_settings({"max_tokens": 0})


def test_repeat_penalty_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Repeat penalty must be positive"):
        generation_settings({"repeat_penalty": 0})


def test_temperature_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="Temperature must be >= 0"):
        generation_settings({"temperature": -1})


def test_top_p_zero_allowed() -> None:
    result = generation_settings({"top_p": 0})
    assert result["top_p"] == 0.0


def test_top_p_one_allowed() -> None:
    result = generation_settings({"top_p": 1})
    assert result["top_p"] == 1.0


def test_top_p_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="Top P must be between 0 and 1"):
        generation_settings({"top_p": 1.1})


def test_min_p_range() -> None:
    result = generation_settings({"min_p": 0})
    assert result["min_p"] == 0.0
    result = generation_settings({"min_p": 1})
    assert result["min_p"] == 1.0
    with pytest.raises(ValueError, match="Min P must be between 0 and 1"):
        generation_settings({"min_p": -0.1})


# ── Defaults ──────────────────────────────────────────────────────────


def test_defaults_are_consistent() -> None:
    result = generation_settings({})
    assert result["temperature"] == 0.1
    assert result["top_k"] == 20
    assert result["top_p"] == 0.9
    assert result["min_p"] == 0.05
    assert result["max_tokens"] == 1024
    assert result["repeat_penalty"] == 1.0
    assert result["presence_penalty"] == 0.0
    assert result["frequency_penalty"] == 0.0


def test_penalties_must_be_finite() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        generation_settings({"presence_penalty": float("inf")})
    with pytest.raises(ValueError, match="must be a finite number"):
        generation_settings({"frequency_penalty": float("nan")})


def test_default_overrides_apply() -> None:
    result = generation_settings(
        {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.01,
            "max_tokens": 2048,
            "repeat_penalty": 1.2,
        }
    )
    assert result["temperature"] == 0.7
    assert result["top_k"] == 40
    assert result["top_p"] == 0.95
    assert result["min_p"] == 0.01
    assert result["max_tokens"] == 2048
    assert result["repeat_penalty"] == 1.2
