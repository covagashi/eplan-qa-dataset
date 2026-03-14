"""
Combina los datasets y muestra estadisticas finales.

Uso:
    python merge_datasets.py
"""
import json
from pathlib import Path
from collections import Counter

JSON_DIR = Path(r"C:\Users\daviann\Documents\_yo\_mantenimiento\Eplan training\json")
FILES = [
    JSON_DIR / "eplan_qa_dataset.jsonl",       # primera pasada
    JSON_DIR / "eplan_qa_enrichment.jsonl",     # segunda pasada (enrichment)
]
OUTPUT = JSON_DIR / "eplan_qa_FINAL.jsonl"


def main():
    all_pairs = []
    seen = set()  # deduplica por instruction

    for f in FILES:
        if not f.exists():
            print(f"SKIP: {f.name} no existe")
            continue

        count = 0
        dupes = 0
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
                key = obj["instruction"].strip().lower()
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)

                # Filtra respuestas demasiado cortas
                if len(obj["output"]) < 20:
                    continue

                all_pairs.append(obj)
                count += 1

        print(f"{f.name}: {count} Q&A validos ({dupes} duplicados eliminados)")

    # Guardar dataset final
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for pair in all_pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Estadisticas
    cats = Counter(p.get("category", "unknown") for p in all_pairs)
    lengths = [len(p["output"]) for p in all_pairs]

    print(f"\n{'='*50}")
    print(f"DATASET FINAL: {len(all_pairs)} pares Q&A")
    print(f"Archivo: {OUTPUT}")
    print(f"\nDistribucion por categoria:")
    for cat, count in cats.most_common():
        print(f"  {cat:25s} {count:5d}  ({count/len(all_pairs)*100:.1f}%)")
    print(f"\nLongitud de respuestas:")
    lengths.sort()
    print(f"  Mediana: {lengths[len(lengths)//2]} chars")
    print(f"  Media:   {sum(lengths)//len(lengths)} chars")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
