"""
Phase 8: Coverage pass — genera Q&A para docs sub-representados en el dataset existente.
Identifica docs con 0 o <2 Q&A y genera pares adicionales.

Uso:
    python generate_qa_coverage.py              # desde el principio
    python generate_qa_coverage.py --resume     # continuar donde se quedo
"""
import os
import json
import time
import sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "Eplan_DOCS"
OUTPUT_DIR = BASE_DIR / "json"
OUTPUT_FILE = OUTPUT_DIR / "eplan_qa_coverage.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress_coverage.json"
BATCH_CHAR_LIMIT = 500_000
MODEL = "gemini-2.0-flash-lite"
MIN_QA_THRESHOLD = 2  # docs con menos de este numero de Q&A se re-procesan

# Archivos JSONL existentes para calcular cobertura
EXISTING_DATASETS = [
    OUTPUT_DIR / "eplan_qa_dataset.jsonl",
    OUTPUT_DIR / "eplan_qa_enrichment.jsonl",
    OUTPUT_DIR / "eplan_qa_code.jsonl",
]

SYSTEM_PROMPT = """You are an expert training dataset generator for fine-tuning LLMs specialized in industrial electrical engineering with EPLAN Electric P8.

## YOUR TASK
From the EPLAN documentation provided, generate high-quality Q&A pairs. These documents have LITTLE OR NO coverage in the existing dataset, so your output is critical for breadth.

## OUTPUT FORMAT
Each line is one JSON. EXACTLY this format:

{"instruction": "question in English", "output": "detailed answer in English", "source": "path/to/file.md", "category": "category"}

Valid categories:
- api_reference (API classes, methods, properties)
- procedural (step by step to achieve something)
- conceptual (what is X, how does Y work)
- troubleshooting (common errors, solutions)
- best_practices (expert recommendations)
- complete_script (if the doc has code, generate code-based Q&A)

## TYPES OF QUESTIONS TO GENERATE (3-6 per document)
Mix these types for variety:
1. Direct factual: "What does X do?" / "What parameters does Y accept?"
2. Procedural: "How do I [achieve X] in EPLAN?"
3. Problem solving: "Why does error X appear when doing Y?"
4. Comparative: "What is the difference between X and Y?"
5. Code-based: "Write a script that uses [class/method from the doc]" (include full code in answer)

## QUALITY RULES
- Answers must be COMPLETE and SELF-SUFFICIENT
- Include exact names of classes, methods, properties when applicable
- If the document contains code, include code snippets in the answer
- Do NOT invent information not present in the source documents
- ALL output in ENGLISH
- Prefer QUALITY over QUANTITY

## PROCESS
Return ONLY plain JSONL text. No markdown formatting, no explanations. Each line one valid JSON.
"""

# --- INIT ---
client = genai.Client(api_key=GEMINI_API_KEY)


def get_covered_sources():
    """Lee datasets existentes y cuenta Q&A por source."""
    source_counts = Counter()
    for dataset_path in EXISTING_DATASETS:
        if not dataset_path.exists():
            continue
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    source = obj.get("source", "")
                    if source:
                        # Normalizar path separators
                        source = source.replace("\\", "/")
                        source_counts[source] += 1
                except json.JSONDecodeError:
                    continue
    return source_counts


def collect_undercovered_files(source_counts):
    """Encuentra docs con cobertura insuficiente."""
    all_files = sorted(DOCS_DIR.rglob("*.md"))
    undercovered = []
    well_covered = 0

    for f in all_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if not text.strip() or len(text) < 100:
            continue

        rel_path = str(f.relative_to(DOCS_DIR)).replace("\\", "/")
        qa_count = source_counts.get(rel_path, 0)

        if qa_count < MIN_QA_THRESHOLD:
            undercovered.append(f)
        else:
            well_covered += 1

    print(f"Docs sub-cubiertos (<{MIN_QA_THRESHOLD} Q&A): {len(undercovered)}")
    print(f"Docs bien cubiertos: {well_covered}")
    return undercovered


def make_batches(files):
    batches = []
    current_batch = []
    current_size = 0

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
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
                    "temperature": 0.3,
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
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                qa_pairs.append(obj)
            else:
                bad_lines += 1
        except json.JSONDecodeError:
            bad_lines += 1

    print(f"  -> {len(qa_pairs)} Q&A validos", end="")
    if bad_lines:
        print(f" ({bad_lines} descartadas)", end="")
    print()

    return qa_pairs


def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in .env file")
        return

    resume = "--resume" in sys.argv

    # Paso 1: analizar cobertura existente
    print("Analizando cobertura del dataset existente...")
    source_counts = get_covered_sources()
    total_existing = sum(source_counts.values())
    print(f"Q&A existentes: {total_existing} cubriendo {len(source_counts)} docs unicos")

    # Paso 2: encontrar docs sub-cubiertos
    files = collect_undercovered_files(source_counts)
    if not files:
        print("Todos los docs estan bien cubiertos. Nada que hacer.")
        return

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
    print(f"COVERAGE PASS TERMINADO")
    print(f"  Q&A nuevos generados: {progress['total_qa']}")
    print(f"  Archivo: {OUTPUT_FILE}")
    if progress["errors"]:
        print(f"  Batches con error: {len(progress['errors'])}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
