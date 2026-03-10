# EPLAN Q&A Dataset & Fine-Tuning Pipeline

## Goal
Build a fine-tuned LLM specialized in EPLAN Electric P8 (industrial electrical engineering CAD).
Source: 17,289 documentation files (API Reference + User Guide) scraped from EPLAN help.

## Project Structure
```
Eplan training/
├── Eplan_DOCS/              # Source docs (17K .md files, NOT in git)
├── json/
│   └── eplan_qa_FINAL.jsonl # Merged & deduplicated dataset (4,288 Q&A pairs)
├── generate_qa.py           # Pass 1: general Q&A generation via Gemini API
├── generate_qa_enrichment.py# Pass 2: troubleshooting/procedural/conceptual focus
├── merge_datasets.py        # Combine + deduplicate + stats
└── CLAUDE.md
```

## Dataset Stats
- **4,288 Q&A pairs** in English, JSONL format
- Categories: api_reference (82%), procedural (7%), conceptual (6%), troubleshooting (4%), best_practices (1.5%)
- Generated with `gemini-2.0-flash-lite`, temperature 0.2-0.4
- Format: `{"instruction": "...", "output": "...", "source": "...", "category": "..."}`

## Scripts
- All scripts use `google-genai` SDK. API key via env var `GEMINI_API_KEY` or hardcoded
- Batch processing: 500K chars per API call, 116 batches total
- Resumable: `--resume` flag continues from last completed batch
- Rate limiting: 4s between calls

## Roadmap
- [x] Phase 1: Scrape EPLAN documentation (17,289 files)
- [x] Phase 2: Generate base Q&A dataset (4,013 pairs)
- [x] Phase 3: Enrichment pass - procedural/troubleshooting (454 pairs)
- [x] Phase 4: Merge & deduplicate (4,288 final pairs)
- [x] Phase 5: Upload dataset to Hugging Face Hub
- [x] Phase 6: First SFT run — Qwen 2.5 3B on Kaggle T4 (QLoRA, no Unsloth)
- [ ] Phase 7: Code generation pass (~500-1000 pairs) — see ROADMAP.md
- [ ] Phase 8: Retrain with code-enriched dataset v2
- [ ] Phase 9: Convert to GGUF for Ollama
- [ ] Phase 10: Integrate with EPLAN MCP server (RAG + fine-tuned model)

## Tech Stack
- Python 3.12 on Windows 10
- Gemini API (free tier) for Q&A generation
- Fine-tune: HuggingFace TRL (SFT) + PEFT (QLoRA) on Kaggle T4 x2
- Base model: Qwen 2.5 3B Instruct
- Target inference: Ollama (local GGUF)
- Integration: existing MCP server at `Eplan_2026_IA_MCP_scripts/`

## Notes
- Eplan_DOCS/ is NOT committed (proprietary + 101MB)
- Only the final dataset (eplan_qa_FINAL.jsonl) is tracked in git
- API keys must NEVER be committed — use GEMINI_API_KEY env var
