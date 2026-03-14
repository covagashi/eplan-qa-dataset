"""
Phase 7: Generar Q&A de codigo — scripts completos, fix-this-code, workflows multi-step.
Solo procesa docs de API Reference + User Guide que contengan codigo.

Uso:
    python generate_qa_code.py              # desde el principio
    python generate_qa_code.py --resume     # continuar donde se quedo
"""
import os
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "Eplan_DOCS"
OUTPUT_DIR = BASE_DIR / "json"
OUTPUT_FILE = OUTPUT_DIR / "eplan_qa_code.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress_code.json"
BATCH_CHAR_LIMIT = 500_000
MODEL = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """You are an expert EPLAN Electric P8 developer and training dataset generator.

## YOUR TASK
From the EPLAN documentation provided, generate CODE-FOCUSED Q&A pairs for fine-tuning an LLM.
Every answer MUST contain working C# code. This is a CODE GENERATION dataset.

## Q&A TYPES TO GENERATE (3-6 per document fragment)

1. **COMPLETE_SCRIPT** (40% of output):
   Full, runnable C# scripts that solve a specific EPLAN automation task.
   - Question: "Write a complete C# script that [does X] using the EPLAN API"
   - Answer MUST include:
     * All required `using Eplan.EplApi.*` imports
     * The class with [Start] attribute or IEplAction implementation
     * Complete method body (NO placeholders, NO `// TODO`)
     * Try/catch error handling
     * Inline comments explaining each API call

2. **FIX_THIS_CODE** (25% of output):
   Present buggy EPLAN code and ask to fix it.
   - Question: Include the FULL buggy code and describe the symptom (e.g., "throws NullReferenceException", "hangs forever", "exports empty PDF")
   - Answer: Explain the root cause, then provide the FULL corrected code with the fix highlighted in comments

3. **MULTI_STEP_WORKFLOW** (20% of output):
   Complex scripts that chain multiple EPLAN API operations.
   - Question: "Create an EPLAN script that: 1) opens project, 2) iterates pages, 3) checks conditions, 4) exports results"
   - Answer: Complete script with clear step separation in comments

4. **CODE_EXPLANATION** (15% of output):
   Present working EPLAN code and ask for explanation.
   - Question: Include the FULL code and ask "Explain what this EPLAN script does step by step"
   - Answer: Line-by-line or block-by-block explanation of the logic and API usage

## CRITICAL CODE QUALITY RULES
- ONLY use classes, methods, and properties that appear in the source documentation
- Do NOT invent API methods that don't exist
- Every script MUST compile if the EPLAN API assemblies are referenced
- Use correct EPLAN namespaces: Eplan.EplApi.Base, Eplan.EplApi.DataModel, Eplan.EplApi.HEServices, etc.
- Include `using System;` and other standard .NET imports as needed
- Use proper C# conventions (PascalCase methods, camelCase locals)
- For fix_this_code: the bug must be realistic (null ref, wrong method signature, missing dispose, wrong enum value)

## OUTPUT FORMAT
Each line is one JSON. EXACTLY this format:

{"instruction": "question in English", "output": "detailed answer with code in English", "source": "path/to/file.md", "category": "complete_script|fix_this_code|multi_step_workflow|code_explanation"}

## RULES
- ALL output in ENGLISH
- Return ONLY plain JSONL text. No markdown formatting, no explanations outside JSON.
- Each line one valid JSON.
- Do NOT generate non-code Q&A (those exist already). EVERY answer must have code.
"""

# --- INIT ---
client = genai.Client(api_key=GEMINI_API_KEY)


def has_code_content(text):
    """Filtra docs que probablemente contienen info de API/codigo."""
    code_indicators = [
        "```", "public class", "public void", "public static",
        "using Eplan", "namespace Eplan", "IEplAction", "IEplAddIn",
        "[Start]", "void ", "get;", "set;", "{get;", "property ",
        "Parameter", "Return", "Exception", "Method", "Constructor",
        "Overload", "Event", "Delegate", "Interface", "Enum",
        "Action", "class ", "struct ", "abstract ", "virtual ",
    ]
    return any(indicator in text for indicator in code_indicators)


def collect_files():
    """Recopila .md que contengan contenido de API/codigo."""
    all_files = sorted(DOCS_DIR.rglob("*.md"))
    code_files = []
    skipped = 0

    for f in all_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not text.strip() or len(text) < 50:
            continue
        if has_code_content(text):
            code_files.append(f)
        else:
            skipped += 1

    print(f"Archivos con codigo/API: {len(code_files)} (saltados: {skipped})")
    return code_files


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


VALID_CATEGORIES = {"complete_script", "fix_this_code", "multi_step_workflow", "code_explanation"}


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
                    "temperature": 0.4,
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
    no_code = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "instruction" not in obj or "output" not in obj:
                bad_lines += 1
                continue

            # Validar que la respuesta contiene codigo
            output = obj["output"]
            if "using " not in output and "class " not in output and "void " not in output and "```" not in output:
                no_code += 1
                continue

            # Normalizar categoria
            cat = obj.get("category", "complete_script")
            if cat not in VALID_CATEGORIES:
                cat = "complete_script"
            obj["category"] = cat

            qa_pairs.append(obj)
        except json.JSONDecodeError:
            bad_lines += 1

    print(f"  -> {len(qa_pairs)} Q&A validos", end="")
    if bad_lines:
        print(f" ({bad_lines} descartadas)", end="")
    if no_code:
        print(f" ({no_code} sin codigo filtradas)", end="")
    print()

    return qa_pairs


def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in .env file")
        return

    resume = "--resume" in sys.argv

    files = collect_files()
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
    print(f"CODE GENERATION TERMINADO")
    print(f"  Q&A de codigo generados: {progress['total_qa']}")
    print(f"  Archivo: {OUTPUT_FILE}")
    if progress["errors"]:
        print(f"  Batches con error: {len(progress['errors'])}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
