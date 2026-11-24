"""OpenAI-compatible VLM client, config loading and tolerant JSON parsing.

Client conventions follow the NVIDIA Retail Catalog Enrichment blueprint (OpenAI SDK, NGC_API_KEY,
chat_template_kwargs.enable_thinking). `parse_llm_json` is adapted from that blueprint
(src/backend/utils.py, Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger("waymo_enrich.llm")

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    vlm = cfg.setdefault("vlm", {})
    vlm["base_url"] = os.getenv("VLM_BASE_URL") or vlm.get("base_url")
    vlm["model"] = os.getenv("VLM_MODEL") or vlm.get("model")
    llm = cfg.setdefault("llm", {})  # text-only structuring model; inherits endpoint/sampling from vlm
    for k in ("base_url", "temperature", "top_p", "max_tokens", "enable_thinking", "retries"):
        llm.setdefault(k, vlm.get(k))
    llm["model"] = os.getenv("LLM_MODEL") or llm.get("model")
    cfg.setdefault("ego_motion", {"stopped_speed_mps": 0.5, "accel_threshold_mps2": 1.0})
    return cfg


def parse_llm_json(text: str, *, extract_braces: bool = True, strip_comments: bool = True) -> Optional[dict]:
    """Parse a JSON object from an LLM response (handles think tags, code fences, comments)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
                break
    if extract_braces:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            text = text[a : b + 1]
    if strip_comments:
        text = re.sub(r"//.*?(?=\n|$)", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def image_part(jpeg_bytes: bytes) -> dict:
    b64 = base64.b64encode(jpeg_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def resolve_api_key(base_url: str) -> str:
    """Pick the key that matches the endpoint: NEBIUS_API_KEY for Nebius Token Factory, else NGC_API_KEY."""
    if "nebius" in base_url:
        key, name = os.getenv("NEBIUS_API_KEY"), "NEBIUS_API_KEY"
    else:
        key, name = os.getenv("NGC_API_KEY") or os.getenv("NVIDIA_API_KEY"), "NGC_API_KEY"
    if not key:
        raise RuntimeError(f"{name} is not set (see .env.example)")
    return key


class VLMClient:
    """Thin wrapper: chat(content_parts, system) -> text."""

    def __init__(self, cfg: dict[str, Any], section: str = "vlm"):
        from openai import OpenAI  # imported lazily so tests/mock runs need no key

        v = cfg[section]
        api_key = resolve_api_key(v.get("base_url") or "")
        self.cfg = v
        self.client = OpenAI(base_url=v["base_url"], api_key=api_key)

    def chat(self, parts: list[dict], system: Optional[str] = None) -> str:
        v = self.cfg
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": parts}]
        kwargs: dict[str, Any] = dict(
            model=v["model"], messages=messages, temperature=v.get("temperature", 0.1),
            top_p=v.get("top_p", 0.9), max_tokens=v.get("max_tokens", 1024),
        )
        if v.get("enable_thinking") is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(v["enable_thinking"])}}
        completion = self.client.chat.completions.create(**kwargs)
        return (completion.choices[0].message.content or "").strip()


class MockClient:
    """Offline stand-in for tests and dry runs. `fn(parts, system) -> str`."""

    def __init__(self, fn: Optional[Callable[[list[dict], Optional[str]], str]] = None):
        self.fn = fn or (lambda parts, system: json.dumps({
            "time_of_day": "day", "weather": "dry", "road_type": "urban_street",
            "pedestrian_visible": False, "cyclist_visible": False, "ego_motion": "cruising", "notable_event": None,
        }))
        self.calls: list[tuple[list[dict], Optional[str]]] = []

    def chat(self, parts: list[dict], system: Optional[str] = None) -> str:
        self.calls.append((parts, system))
        return self.fn(parts, system)
