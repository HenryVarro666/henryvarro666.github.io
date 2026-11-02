---
title: "Medical Multi-Agent System"
date: 2026-10-26
repo: "https://github.com/HenryVarro666/Medical-Multi-Agent-System"
lang: Python
stack: ["LangGraph", "FastAPI", "Pydantic", "Docker"]
status: "research prototype"
topics: [llm, healthcare]
---
Python research prototype using LangGraph and FastAPI for intake, diagnosis, treatment, coding, and a deterministic audit. Extends an open-source scaffold with symptom-based retrieval, curated drug-interaction and allergy checks, bounded diagnosis retries, and an OpenAI-compatible or Codex CLI model backend. Coding receives the full patient record and every proposed ICD-10 code is checked against a packaged official FY2027 catalog of 74,879 complete codes. Includes offline regression tests and an evaluation harness with versioned synthetic cases, archived stage outputs and a two-model comparison.
