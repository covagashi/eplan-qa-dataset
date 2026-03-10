"""
Pipeline para generar Q&A con Gemini a partir de docs EPLAN.
Ajusta GEMINI_API_KEY antes de ejecutar.

Uso:
    python generate_qa.py              # procesa todo desde el principio
    python generate_qa.py --resume     # continua desde el ultimo batch completado
"""
import os
import json
import time
import sys
from pathlib import Path
from google import genai

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DOCS_DIR = Path(r"C:\Uplan training\Eplan_DOCS")
OUTPUT_DIR = Path(r"C:\Eplan training\json")
OUTPUT_FILE = OUTPUT_DIR / "eplan_qa_dataset.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
BATCH_CHAR_LIMIT = 500_000
MODEL = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """You are an expert training dataset generator for fine-tuning LLMs specialized in industrial electrical engineering with EPLAN Electric P8.

## YOUR TASK
From the EPLAN documentation provided, generate high-quality Question-Answer (Q&A) pairs in English in JSONL format.
These pairs will be used to fine-tune a language model that acts as an EPLAN technical assistant.

## OUTPUT FORMAT
Each line is an independent JSON. Use EXACTLY this format, respond in ENGLISH:

{"instruction": "question here in English", "output": "answer here in English", "source": "path/to/file.md", "category": "category"}

Valid categories:
- api_reference (API classes, methods, properties)
- user_guide (workflows, configuration, general usage)
- troubleshooting (common errors, solutions)
- conceptual (what is X, how does Y work, differences between)
- procedural (step by step to achieve something)
- masterdata (master data, articles, macros)

## TYPES OF QUESTIONS TO GENERATE
For each documentation fragment, generate between 3 and 8 varied Q&A pairs. Mix these types:
1. Direct factual: "What does class X do?" / "What parameters does method Y accept?"
2. Procedural: "How do I create a new schematic in EPLAN?"
3. Problem solving: "Why does error X appear when doing Y?"
4. Comparative: "What is the difference between X and Y in EPLAN?"
5. Best practices: "What is the recommended way to...?"
6. Integration/API: "How do I programmatically access...?"

## QUALITY RULES
- Answers must be COMPLETE and SELF-SUFFICIENT (never say "see documentation").
- Include exact names of classes, methods, properties, actions when applicable.
- If the document contains code, include code snippets in the answer.
- Questions must be realistic, as a real electrical engineer would ask them.
- Do NOT generate trivial questions like "What is EPLAN?".
- All Q&A must be 100% in ENGLISH.
- Do NOT invent information that is not in the source document.
- Prefer QUALITY over QUANTITY. 3 excellent pairs are better than 8 mediocre ones.

## PROCESS
Return ONLY plain JSONL text. No markdown formatting (no ```), no additional explanations. Each line one valid JSON.
"""

# --- INIT ---
client = genai.Client(api_key=GEMINI_API_KEY)


def collect_files():
    """Recopila todos los .md y .csv."""
    files = sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.csv"))
    return files


def make_batches(files):
    """Agrupa archivos en batches que no excedan el limite de caracteres."""
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
    """Carga el progreso guardado."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_batch": 0, "total_qa": 0, "errors": []}


def save_progress(progress):
    """Guarda el progreso."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def process_batch(batch_content, batch_num, total_batches):
    """Envia un batch a Gemini y parsea el JSONL resultante."""
    joined = "".join(batch_content)
    print(f"\n[Batch {batch_num}/{total_batches}] Enviando {len(joined):,} chars...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"{SYSTEM_PROMPT}\n\n{joined}",
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 65536,
                }
            )
            text = response.text.strip()
            break
        except Exception as e:
            print(f"  Intento {attempt + 1}/{max_retries} fallo: {e}")
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Esperando {wait}s antes de reintentar...")
                time.sleep(wait)
            else:
                print(f"  -> FALLO DEFINITIVO en batch {batch_num}")
                return []

    # Limpiar posible markdown que Gemini a veces devuelve
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else 3
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Parsear lineas JSONL validas
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
        print(f" ({bad_lines} lineas descartadas)", end="")
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

                # Rate limiting
                if i < len(batches):
                    time.sleep(4)

            except Exception as e:
                print(f"  ERROR en batch {i}: {e}")
                progress["errors"].append({"batch": i, "error": str(e)})
                save_progress(progress)
                time.sleep(30)

    print(f"\n{'='*50}")
    print(f"TERMINADO")
    print(f"  Total Q&A generados: {progress['total_qa']}")
    print(f"  Archivo: {OUTPUT_FILE}")
    if progress["errors"]:
        print(f"  Batches con error: {len(progress['errors'])}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
