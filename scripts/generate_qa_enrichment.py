"""
Segunda pasada: enriquecer dataset con Q&A procedurales, troubleshooting y conceptuales.
Reutiliza los mismos docs pero con prompt que PROHIBE api_reference factuales.

Uso:
    python generate_qa_enrichment.py              # desde el principio
    python generate_qa_enrichment.py --resume      # continuar donde se quedo
"""
import os
import json
import time
import sys
from pathlib import Path
from google import genai

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "Eplan_DOCS"
OUTPUT_DIR = BASE_DIR / "json"
OUTPUT_FILE = OUTPUT_DIR / "eplan_qa_enrichment.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress_enrichment.json"
BATCH_CHAR_LIMIT = 500_000
MODEL = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """You are an expert training dataset generator for fine-tuning LLMs specialized in industrial electrical engineering with EPLAN Electric P8.

## YOUR TASK
From the EPLAN documentation provided, generate Q&A pairs focused EXCLUSIVELY on practical usage, problem-solving, and deeper understanding.

IMPORTANT: Do NOT generate simple factual questions like "What does class X do?" or "What parameters does method Y accept?". Those already exist in the dataset. Instead, generate ONLY these types:

## REQUIRED Q&A TYPES (generate 3-6 per document fragment)

1. **TROUBLESHOOTING** (40% of output): Real problems an engineer would face.
   - "Why does my script throw an exception when trying to access project properties after closing the project?"
   - "I'm getting a NullReferenceException when iterating over pages. What could cause this?"
   - "My export action hangs without producing output. How do I diagnose this?"

2. **PROCEDURAL / HOW-TO** (30% of output): Step-by-step workflows combining multiple API calls.
   - "How do I create a complete workflow to export all pages of a project to PDF programmatically?"
   - "What is the correct sequence of API calls to create a new device, place it on a page, and assign an article number?"
   - "How do I automate batch processing of multiple EPLAN projects?"

3. **CONCEPTUAL / ARCHITECTURAL** (20% of output): Understanding the big picture.
   - "What is the relationship between Project, Page, and Placement in the EPLAN object model?"
   - "How does EPLAN's transaction model work and why should I use it?"
   - "What is the difference between a Function and a FunctionBase in the EPLAN API?"

4. **BEST PRACTICES** (10% of output): Expert-level advice.
   - "What are the best practices for error handling in EPLAN API scripts?"
   - "How should I structure a large EPLAN automation script for maintainability?"
   - "When should I use actions vs direct API calls in EPLAN automation?"

## OUTPUT FORMAT
Each line is one JSON. EXACTLY this format:

{"instruction": "question in English", "output": "detailed answer in English", "source": "path/to/file.md", "category": "category"}

Valid categories: troubleshooting, procedural, conceptual, best_practices

## QUALITY RULES
- Answers must be DETAILED (minimum 3-4 sentences, include code examples when relevant)
- Questions must sound like a real engineer asking for help, not a textbook quiz
- Answers should explain the WHY, not just the WHAT
- Include common pitfalls and edge cases in troubleshooting answers
- For procedural answers, include complete code examples when possible
- Do NOT generate api_reference category — that's already covered
- Do NOT invent information not present in the source documents
- ALL output in ENGLISH

## PROCESS
Return ONLY plain JSONL text. No markdown formatting, no explanations. Each line one valid JSON.
"""

# --- INIT ---
client = genai.Client(api_key=GEMINI_API_KEY)


def collect_files():
    files = sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.csv"))
    return files


def make_batches(files):
    batches = []
    current_batch = []
    current_size = 0

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if not text.strip() or len(text) < 50:
            continue

        rel_path = str(f.relative_to(DOCS_DIR)).replace("\\", "/")
        entry = f"\n\n===== FILE: {rel_path} =====\n{text}"
        entry_size = len(entry)

        if current_size + entry_size > BATCH_CHAR_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(entry)
        current_size += entry_size

    if current_batch:
        batches.append(current_batch)

    return batches


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_batch": 0, "total_qa": 0, "errors": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def process_batch(batch_content, batch_num, total_batches):
    joined = "".join(batch_content)
    print(f"\n[Batch {batch_num}/{total_batches}] Enviando {len(joined):,} chars...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"{SYSTEM_PROMPT}\n\n{joined}",
                config={
                    "temperature": 0.4,  # Un poco mas creativo para troubleshooting
                    "max_output_tokens": 65536,
                }
            )
            text = response.text.strip()
            break
        except Exception as e:
            print(f"  Intento {attempt + 1}/{max_retries} fallo: {e}")
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"  -> FALLO DEFINITIVO en batch {batch_num}")
                return []

    # Limpiar markdown
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Parsear JSONL
    qa_pairs = []
    bad_lines = 0
    api_ref_filtered = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                # Filtrar si Gemini ignora el prompt y genera api_reference
                if obj.get("category") == "api_reference":
                    api_ref_filtered += 1
                    continue
                qa_pairs.append(obj)
            else:
                bad_lines += 1
        except json.JSONDecodeError:
            bad_lines += 1

    print(f"  -> {len(qa_pairs)} Q&A validos", end="")
    if bad_lines:
        print(f" ({bad_lines} descartadas)", end="")
    if api_ref_filtered:
        print(f" ({api_ref_filtered} api_reference filtradas)", end="")
    print()

    return qa_pairs


def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        return

    resume = "--resume" in sys.argv

    files = collect_files()
    print(f"Archivos encontrados: {len(files)}")

    batches = make_batches(files)
    print(f"Batches a procesar: {len(batches)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress() if resume else {"last_batch": 0, "total_qa": 0, "errors": []}
    start_batch = progress["last_batch"]

    if resume and start_batch > 0:
        print(f"Reanudando desde batch {start_batch + 1} ({progress['total_qa']} Q&A previos)")

    mode = "a" if resume and start_batch > 0 else "w"

    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:
        for i, batch in enumerate(batches, 1):
            if i <= start_batch:
                continue

            try:
                pairs = process_batch(batch, i, len(batches))
                for pair in pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                progress["total_qa"] += len(pairs)
                progress["last_batch"] = i
                out.flush()
                save_progress(progress)

                if i < len(batches):
                    time.sleep(4)

            except Exception as e:
                print(f"  ERROR en batch {i}: {e}")
                progress["errors"].append({"batch": i, "error": str(e)})
                save_progress(progress)
                time.sleep(30)

    print(f"\n{'='*50}")
    print(f"ENRICHMENT TERMINADO")
    print(f"  Q&A nuevos generados: {progress['total_qa']}")
    print(f"  Archivo: {OUTPUT_FILE}")
    if progress["errors"]:
        print(f"  Batches con error: {len(progress['errors'])}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
