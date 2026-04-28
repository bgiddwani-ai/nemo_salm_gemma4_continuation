# file: multiprompt_data.py
"""
Convert a flat manifest into multimodal_conversation JSONL for NeMo speechlm2 SALM.
Randomly assigns one of N prompts to each sample.

Schema produced (one JSON object per line):
{
  "id": "...",
  "conversations": [
    {"from": "user",      "value": "<prompt text>",   "type": "text"},
    {"from": "user",      "value": "/path/audio.wav", "duration": 6.42, "type": "audio"},
    {"from": "assistant", "value": "<target text>",   "type": "text"}
  ]
}
"""

import json
import random
import sys
from pathlib import Path

import soundfile as sf


# ---- Edit this list to add / change prompts ----
PROMPTS = [
    "Transcribe the following:",
    "Write down what is said in this recording:",
]

SEED = 42


def get_duration(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate


def convert(src: str, dst: str) -> tuple[int, dict]:
    rng = random.Random(SEED)
    counts = {p: 0 for p in PROMPTS}
    n = 0

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    with open(src) as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            audio_path = rec["audio_filepath"]
            answer     = rec.get("answer") or rec.get("text") or ""
            duration   = rec.get("duration")
            offset     = rec.get("offset")
            uid        = rec.get("id", f"utt_{i:08d}")

            if duration is None:
                duration = get_duration(audio_path)

            # ---- pick a random prompt per sample ----
            prompt = rng.choice(PROMPTS)
            counts[prompt] += 1

            audio_turn = {
                "from": "user",
                "value": audio_path,
                "duration": float(duration),
                "type": "audio",
            }
            if offset is not None:
                audio_turn["offset"] = float(offset)

            out = {
                "id": uid,
                "conversations": [
                    {"from": "user",      "value": prompt, "type": "text"},
                    audio_turn,
                    {"from": "assistant", "value": answer, "type": "text"},
                ],
            }

            for k in ("lang", "task", "pnc"):
                if k in rec:
                    out[k] = rec[k]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    return n, counts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nemo_to_multiprompt_manifest.py <input_manifest.jsonl> <output_manifest.jsonl>")
        sys.exit(1)

    count, distribution = convert(sys.argv[1], sys.argv[2])
    print(f"Wrote {count} records to {sys.argv[2]}")
    print("Prompt distribution:")
    for prompt, c in distribution.items():
        pct = 100 * c / count if count else 0
        print(f"  [{c:>6}] ({pct:5.2f}%)  {prompt!r}")