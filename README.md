# EPLAN Electric P8 — Q&A Dataset for Fine-Tuning

Fine-tuning dataset generated from 17,289 EPLAN Electric P8 documentation files. Designed to train a specialized LLM assistant for industrial electrical engineering with EPLAN.

## Dataset

**4,288 Q&A pairs** in `json/eplan_qa_FINAL.jsonl`

| Category | Count | % |
|----------|-------|---|
| API Reference | 3,514 | 82% |
| Procedural | 308 | 7% |
| Conceptual | 242 | 6% |
| Troubleshooting | 156 | 4% |
| Best Practices | 65 | 1.5% |

### Format (JSONL)
```json
{"instruction": "How do I export all pages of a project to PDF programmatically?", "output": "To export all pages to PDF...", "source": "API Reference/Actions/export.md", "category": "procedural"}
```

## Pipeline

```
EPLAN Docs (17K files) → Gemini API → Q&A Generation → Merge & Deduplicate → Fine-Tune
```

| Script | Purpose |
|--------|---------|
| `generate_qa.py` | Base Q&A generation (api_reference focus) |
| `generate_qa_enrichment.py` | Enrichment pass (troubleshooting, procedural, conceptual) |
| `merge_datasets.py` | Combine, deduplicate, and generate stats |

### Usage
```bash
# Set your API key
export GEMINI_API_KEY="your-key-here"

# Generate base dataset
python generate_qa.py

# Generate enrichment (procedural/troubleshooting focus)
python generate_qa_enrichment.py

# Merge into final dataset
python merge_datasets.py
```

All scripts support `--resume` to continue from the last completed batch.

## Roadmap

- [x] Scrape EPLAN documentation
- [x] Generate base Q&A dataset
- [x] Enrichment pass
- [x] Merge & deduplicate
- [ ] Upload dataset to Hugging Face Hub
- [ ] Fine-tune (SFT with LoRA) on a 7B model
- [ ] Convert to GGUF for local inference (Ollama)
- [ ] Integrate with [EPLAN MCP Server](https://github.com/covagashi/Eplan_2026_IA_MCP_scripts)

## Requirements
```
pip install google-genai
```

## License
Dataset generated from EPLAN public documentation. Scripts are MIT.
