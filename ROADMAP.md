# EPLAN Fine-Tuning Roadmap

## Current Status

| Phase | Status | Output |
|-------|--------|--------|
| 1. Scrape docs | Done | 17,289 .md files |
| 2. Base Q&A generation | Done | 4,013 pairs (`generate_qa.py`) |
| 3. Enrichment pass | Done | 454 pairs (`generate_qa_enrichment.py`) |
| 4. Merge & deduplicate | Done | 4,288 pairs (`eplan_qa_FINAL.jsonl`) |
| 5. Upload to HuggingFace | Done | `covaga/eplan-qa-dataset` |
| 6. First SFT run | Done | Qwen 2.5 3B, Kaggle T4, QLoRA |
| **7. Code generation pass** | **Next** | ~500-1000 code pairs (`generate_qa_code.py`) |
| **8. Coverage pass** | **Next** | ~500-1000 new pairs from under-covered docs (`generate_qa_coverage.py`) |
| 9. Merge into v2 | Pending | `eplan_qa_v2_FINAL.jsonl` (~5,500-6,300 pairs) |
| 10. Retrain with v2 dataset | Pending | Second SFT run + loss curve plot |
| 11. GGUF export | Pending | Ollama-ready model |
| 12. MCP integration | Pending | RAG + fine-tuned model |

---

## Phase 7: Code Generation Pass

### Problem
The v1 model understands EPLAN domain and terminology but generates code with issues:
- Incomplete scripts (missing `using` statements, no error handling)
- Invented API methods that don't exist
- Shallow code explanations

Current dataset is 82% `api_reference` (descriptive) — the model knows *what* APIs do but not *how to write working code*.

### Goal
Generate 500-1000 code-focused Q&A pairs to teach the model to write correct, complete EPLAN C# scripts.

### Sub-categories

| Type | Target % | Example Instruction |
|------|----------|-------------------|
| `complete_script` | 40% | "Write a complete C# script that exports all project pages to individual PDFs using EPLAN API" |
| `fix_this_code` | 25% | "This EPLAN script throws NullReferenceException when iterating pages: [buggy code]. Find and fix the bug." |
| `multi_step_workflow` | 20% | "Create an EPLAN automation script that: 1) opens a project, 2) iterates all pages, 3) checks for missing device tags, 4) exports a report" |
| `code_explanation` | 15% | "Explain what this EPLAN API script does step by step: [working code]" |

### Source Docs to Prioritize
1. **API Reference** — classes that have code examples in the documentation
2. **Scripting/Automation section** — User Guide chapters on EPLAN scripting
3. **Action Reference** — built-in EPLAN actions with parameters (export, import, print, etc.)
4. **Examples folder** — if any sample scripts exist in the docs

### Prompt Strategy for Gemini

Key rules for the generation prompt:
- Every `complete_script` answer MUST be a full, runnable C# script
- All scripts MUST include proper `using Eplan.EplApi.*` imports
- All scripts MUST include try/catch error handling
- All scripts MUST include comments explaining each API call
- `fix_this_code` pairs MUST include both the buggy version (in instruction) and the corrected version (in output)
- No placeholder code (`// TODO`, `// implement here`) — complete implementations only
- No inventing APIs — only use classes/methods that appear in the source docs

### Script: `generate_qa_code.py`

Same pipeline as `generate_qa_enrichment.py` but with:
- New prompt focused exclusively on code generation
- Filter: only process API Reference docs (skip User Guide for this pass)
- Higher temperature (0.4) for more diverse code patterns
- Post-processing: validate every output has `using` statements and code blocks
- Output: `json/eplan_qa_code.jsonl`

---

## Phase 8: Coverage Pass

### Problem
17,289 docs but only 4,288 Q&A → ~0.25 Q&A per doc. Many docs got 0 questions (skipped by batching, too short, or low-priority in Pass 1).

### Goal
Identify under-covered docs and generate additional Q&A to improve breadth.

### Strategy
1. **Detect gaps**: cross-reference `source` field in existing JSONL with all files in `Eplan_DOCS/`
2. **Filter**: docs with 0 Q&A, or docs >500 chars that only produced 1 Q&A
3. **Generate**: same general prompt as Pass 1 but ONLY on uncovered/under-covered docs
4. **Target**: ~500-1000 new pairs across all categories

### Script: `generate_qa_coverage.py`

- Read existing JSONL to build set of covered `source` paths
- Only process docs NOT in that set (or with <2 Q&A)
- Same prompt as `generate_qa.py` (all categories welcome)
- Output: `json/eplan_qa_coverage.jsonl`

---

## Phase 9: Merge into v2

1. Merge all 4 JSONL files: base + enrichment + code + coverage
2. Deduplicate by instruction similarity
3. Output: `json/eplan_qa_v2_FINAL.jsonl`
4. Expected total: ~5,500-6,300 pairs
5. Upload v2 to HuggingFace Hub

---

## Phase 6 Results: First SFT Run

### Training Log
| Step | Loss |
|------|------|
| 10 | 0.7922 |
| 20 | 0.6844 |
| 30 | 0.6412 |
| 40 | 0.6380 |
| 50 | 0.5510 |
| 60 | 0.5357 |
| 70 | 0.5355 |
| 80 | 0.5013 |
| 90 | 0.4878 |
| 100 | 0.3906 |
| 110 | 0.3885 |
| 120 | 0.4429 |

- **Final avg loss**: 0.5427
- **Runtime**: 4,006s (~67 min)
- **Steps**: 129 (3 epochs)

### Analysis
- **Steps 10-50**: Fast learning (model learns EPLAN domain)
- **Steps 60-110**: Plateau (~0.39-0.53, model converged)
- **Step 120**: Loss rises to 0.44 — early sign of **overfitting**
- **Conclusion**: 3 epochs was slightly too many. 2 epochs is the sweet spot.

### Lessons for Phase 8
- Use `num_train_epochs=2` instead of 3
- Add train/eval split (90/10) to monitor overfitting with eval_loss
- Use `save_strategy="steps"` + `save_steps=40` for intermediate checkpoints
- Keep `per_device_train_batch_size=1` + `gradient_accumulation_steps=8` (T4 VRAM constraint)

---

## Phase 10: Retrain with v2 Dataset

- Same setup: Qwen 2.5 3B, Kaggle T4 x2, QLoRA
- **Changed hyperparameters** based on v1 results:
  - `num_train_epochs=2` (was 3, overfitting detected)
  - `save_strategy="steps"`, `save_steps=40`
  - Train/eval split 90/10
  - `eval_strategy="steps"`, `eval_steps=40`
- **Plot loss curve** after training (train_loss + eval_loss) to verify no overfitting
- Compare v1 vs v2 on:
  - Code completeness (has imports, error handling?)
  - API accuracy (uses real EPLAN methods?)
  - Troubleshooting depth

---

## Phase 11: GGUF Export

Options:
- **Option A**: llama.cpp `convert_hf_to_gguf.py` locally
- **Option B**: HuggingFace Space `gguf-my-repo`
- Target quantization: Q4_K_M (~2GB for 3B model)

---

## Phase 12: MCP Server Integration

Combine fine-tuned model + RAG for maximum accuracy:
- **Fine-tuned model** → knows EPLAN domain, writes code patterns, understands terminology
- **RAG (via MCP server)** → retrieves actual documentation at query time, validates API signatures
- Existing MCP server at `Eplan_2026_IA_MCP_scripts/`

Architecture:
```
User Query → MCP Server → RAG retrieval (docs) → Fine-tuned model (Ollama) → Response
```

This solves the "hallucinated APIs" problem: the model knows the patterns but RAG provides the ground truth.
