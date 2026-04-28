#!/usr/bin/env python3
"""
Convert a NeMo manifest file by adding a `context` field with a randomly
selected prompt from a fixed list.

Input manifest entry (JSON lines):
    {"audio_filepath": "...", "text": "...", "duration": 1.23, "lang": "en"}

Output manifest entry (JSON lines):
    {"audio_filepath": "...", "text": "...", "duration": 1.23, "lang": "en",
     "context": "Transcribe the following:"}

Usage:
    python nemo_add_context.py input.json output.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROMPTS = [
    "Transcribe the following:",
    "Write down what is said in this recording:",
]
SEED = 42


def add_context(input_path: Path, output_path: Path) -> int:
    rng = random.Random(SEED)
    count = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping malformed JSON on line {line_num}: {e}",
                      file=sys.stderr)
                continue

            entry["context"] = rng.choice(PROMPTS)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input NeMo manifest (.json/.jsonl)")
    parser.add_argument("output", type=Path, help="Output manifest with context field")
    args = parser.parse_args()

    if not args.input.is_file():
        sys.exit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = add_context(args.input, args.output)
    print(f"Wrote {n} entries to {args.output}")


if __name__ == "__main__":
    main()