"""
Phase 9: Merge all datasets into v2 — base + enrichment + code + coverage.
Deduplica por instruction y genera estadisticas.

Uso:
    python merge_datasets_v2.py
"""
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "json"

FILES = [
    JSON_DIR / "eplan_qa_dataset.jsonl",       # Phase 2: base
    JSON_DIR / "eplan_qa_enrichment.jsonl",     # Phase 3: enrichment
    JSON_DIR / "eplan_qa_code.jsonl",           # Phase 7: code generation
    JSON_DIR / "eplan_qa_coverage.jsonl",       # Phase 8: coverage
]
OUTPUT = JSON_DIR / "eplan_qa_v2_FINAL.jsonl"


def normalize_instruction(text):
    """Normaliza instruction para deduplicacion mas agresiva."""
    return " ".join(text.strip().lower().split())


def main():
    all_pairs = []
    seen = set()
    file_stats = {}

    for f in FILES:
        if not f.exists():
            print(f"SKIP: {f.name} no existe")
            continue

        count = 0
        dupes = 0
        short = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "instruction" not in obj or "output" not in obj:
                    continue

                # Deduplica
                key = normalize_instruction(obj["instruction"])
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)

                # Filtra respuestas demasiado cortas
                if len(obj["output"]) < 20:
                    short += 1
                    continue

                all_pairs.append(obj)
                count += 1

        file_stats[f.name] = {"added": count, "dupes": dupes, "short": short}
        print(f"{f.name}: {count} Q&A validos ({dupes} dupes, {short} cortos)")

    if not all_pairs:
        print("No hay datos para merge.")
        return

    # Guardar dataset final
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for pair in all_pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Estadisticas
    cats = Counter(p.get("category", "unknown") for p in all_pairs)
    lengths = [len(p["output"]) for p in all_pairs]
    sources = set(p.get("source", "") for p in all_pairs)

    print(f"\n{'='*60}")
    print(f"DATASET v2 FINAL: {len(all_pairs)} pares Q&A")
    print(f"Archivo: {OUTPUT}")
    print(f"Docs unicos referenciados: {len(sources)}")

    print(f"\nPor archivo fuente:")
    for fname, stats in file_stats.items():
        print(f"  {fname:35s} +{stats['added']:5d}")

    print(f"\nDistribucion por categoria:")
    for cat, count in cats.most_common():
        pct = count / len(all_pairs) * 100
        bar = "#" * int(pct / 2)
        print(f"  {cat:25s} {count:5d}  ({pct:5.1f}%) {bar}")

    print(f"\nLongitud de respuestas:")
    lengths.sort()
    print(f"  Min:     {lengths[0]:,} chars")
    print(f"  Mediana: {lengths[len(lengths)//2]:,} chars")
    print(f"  Media:   {sum(lengths)//len(lengths):,} chars")
    print(f"  Max:     {lengths[-1]:,} chars")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
