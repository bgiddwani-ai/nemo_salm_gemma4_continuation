"""
Merge manifest data (audio_filepath, text, duration) into conversation JSON format.

Input manifest format (JSONL - one JSON object per line):
    {"audio_filepath": "/path/to/utt1.wav", "text": "hello world", "duration": 12.34}

Output format:
    {
        "id": "<sample-id>",
        "conversations": [
            {"from": "user",      "value": "<prompt>",   "type": "text"},
            {"from": "user",      "value": "/utt1.wav",  "duration": 1.34, "type": "audio"},
            {"from": "assistant", "value": "hello world", "type": "text"},
            {"from": "user",      "value": "/utt2.wav",  "duration": 2.87, "type": "audio"},
            {"from": "assistant", "value": "bye world", "type": "text"},
            ...
        ]
    }

Constraints:
    - Total duration per conversation <= 30 seconds
    - Prompt randomly chosen from PROMPTS list
"""

import json
import random
import argparse
import uuid
from pathlib import Path

PROMPTS = [
    "Transcribe the following:",
    "Write down what is said in this recording:",
]

MAX_DURATION = 30.0


def load_manifest(manifest_path: str) -> list[dict]:
    """Load a JSONL manifest file."""
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
    return records


def validate_record(record: dict, line_num: int) -> bool:
    """Check that a record has the required fields."""
    required = {"audio_filepath", "text", "duration"}
    missing = required - record.keys()
    if missing:
        print(f"Warning: Record {line_num} missing fields {missing}, skipping.")
        return False
    try:
        float(record["duration"])
    except (TypeError, ValueError):
        print(f"Warning: Record {line_num} has non-numeric duration, skipping.")
        return False
    return True


def group_into_conversations(
    records: list[dict],
    max_duration: float = MAX_DURATION,
) -> list[list[dict]]:
    """
    Greedily group records into conversations where the total
    audio duration does not exceed max_duration seconds.
    """
    groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_duration = 0.0

    for record in records:
        dur = float(record["duration"])

        # A single utterance longer than the limit gets its own group with a warning
        if dur > max_duration:
            print(
                f"Warning: '{record['audio_filepath']}' has duration {dur:.2f}s "
                f"> max {max_duration}s. Placing it in its own conversation."
            )
            if current_group:
                groups.append(current_group)
                current_group = []
                current_duration = 0.0
            groups.append([record])
            continue

        if current_duration + dur > max_duration:
            # Start a new conversation
            groups.append(current_group)
            current_group = [record]
            current_duration = dur
        else:
            current_group.append(record)
            current_duration += dur

    if current_group:
        groups.append(current_group)

    return groups


def build_conversation(group: list[dict], sample_id: str) -> dict:
    """Build a single conversation object from a group of records."""
    prompt = random.choice(PROMPTS)

    turns = [
        {"from": "user", "value": prompt, "type": "text"}
    ]

    for record in group:
        # Audio turn
        turns.append({
            "from": "user",
            "value": record["audio_filepath"],
            "duration": float(record["duration"]),
            "type": "audio",
        })
        # Transcription turn
        turns.append({
            "from": "assistant",
            "value": record["text"],
            "type": "text",
        })

    return {"id": sample_id, "conversations": turns}


def merge_manifest(
    manifest_path: str,
    output_path: str,
    max_duration: float = MAX_DURATION,
    id_prefix: str = "sample",
    use_uuid: bool = False,
) -> None:
    """
    Main entry point: read manifest, group records, write output JSONL.

    Args:
        manifest_path: Path to input JSONL manifest.
        output_path:   Path to write output JSONL.
        max_duration:  Maximum total audio duration per conversation (seconds).
        id_prefix:     Prefix for sequential IDs (ignored when use_uuid=True).
        use_uuid:      Use UUID4 strings instead of sequential IDs.
    """
    records = load_manifest(manifest_path)
    print(f"Loaded {len(records)} records from '{manifest_path}'.")

    # Validate
    valid_records = [r for i, r in enumerate(records) if validate_record(r, i + 1)]
    print(f"{len(valid_records)} valid records after validation.")

    if not valid_records:
        print("No valid records found. Exiting.")
        return

    groups = group_into_conversations(valid_records, max_duration)
    print(f"Grouped into {len(groups)} conversation(s) "
          f"(max duration per conversation: {max_duration}s).")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, group in enumerate(groups):
            sample_id = str(uuid.uuid4()) if use_uuid else f"{id_prefix}_{idx:05d}"
            conversation = build_conversation(group, sample_id)
            out_f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Wrote {len(groups)} conversation(s) to '{output_path}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge audio manifest into multi-turn conversation JSONL."
    )
    parser.add_argument(
        "manifest",
        help="Path to input JSONL manifest (audio_filepath, text, duration).",
    )
    parser.add_argument(
        "output",
        help="Path to write output JSONL file.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION,
        help=f"Max total audio duration per conversation in seconds (default: {MAX_DURATION}).",
    )
    parser.add_argument(
        "--id-prefix",
        default="sample",
        help="Prefix for sequential sample IDs (default: 'sample').",
    )
    parser.add_argument(
        "--use-uuid",
        action="store_true",
        help="Use UUID4 for sample IDs instead of sequential integers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible prompt assignment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    merge_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        max_duration=args.max_duration,
        id_prefix=args.id_prefix,
        use_uuid=args.use_uuid,
    )