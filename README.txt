# SALM Gemma4 Training

Training recipes for SALM (Speech-Augmented Language Model) with Gemma4, using NeMo's `speechlm2` framework with Lhotse-based data loading across multiple dataset formats.

## Setup

Clone the NeMo fork and check out the `salm-continuation-dataset` branch:

```bash
git clone https://github.com/bgiddwani-ai/NeMo.git
cd NeMo
git checkout salm-continuation-dataset
```

Install the package with all optional dependencies and upgrade `transformers`:

```bash
pip install -e ".[all]"
pip install --upgrade transformers
```

## Data preparation

Each variant expects a manifest in a specific shape. The scripts below convert a base NeMo manifest (`input.json`) into the format required by the corresponding training config. The `v1` and `v2.2` variants consume tarred NeMo manifests directly and do not need a conversion step here.

| Script | Used for | Purpose |
|---|---|---|
| `nemo_add_context.py` | v2.1 | Adds context fields to a NeMo manifest |
| `nemo_to_multiprompt_manifest.py` | v2.3 | Converts a NeMo manifest into the multi-prompt conversational format |
| `nemo_to_continuation_multiprompt_manifest.py` | v3.1 | Converts a NeMo manifest into the multi-prompt conversational *continuation* format |

```bash
# v2.1 — multi-prompt NeMo manifest
python nemo_add_context.py input.json output.json

# v2.2 — multi-prompt NeMo manifest
python /home/salm/NeMo/scripts/speech_recognition/convert_to_tarred_audio_dataset.py output.json 

# v2.3 — multi-prompt conversational format
python nemo_to_multiprompt_manifest.py input.json output.json

# v3.1 — multi-prompt conversational continuation format
python nemo_to_continuation_multiprompt_manifest.py input.json output.json
```

## Training

All training runs use `torchrun` with 8 GPUs and the same entry script (`salm_train.py`). Each variant differs only in its config file, which selects the data format and prompting strategy.

### Common arguments

- **Script:** `/home/salm/NeMo/examples/speechlm2/salm_train.py`
- **Config path:** `/home/salm/conf`
- **Launcher:** `torchrun --nproc-per-node=8`

### v1 — Lhotse + NeMo tarred dataset

Single-prompt training on a tarred NeMo manifest read through Lhotse.

```bash
torchrun --nproc-per-node=8 /home/salm/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/salm/conf \
    --config-name=salm_gemma4_v1
```

### v2.1 — Lhotse + multi-prompt NeMo manifest

Multi-prompt training on a standard (non-tarred) NeMo manifest.

```bash
torchrun --nproc-per-node=8 /home/salm/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/salm/conf \
    --config-name=salm_gemma4_v2_1
```

### v2.2 — Lhotse + multi-prompt NeMo tarred dataset

Multi-prompt training on a tarred NeMo manifest.

```bash
torchrun --nproc-per-node=8 /home/salm/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/salm/conf \
    --config-name=salm_gemma4_v2_2
```

### v2.3 — Lhotse + multi-prompt conversational format

Multi-prompt training on a conversational-format dataset.

```bash
torchrun --nproc-per-node=8 /home/salm/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/salm/conf \
    --config-name=salm_gemma4_v2_3
```

### v3.1 — Lhotse + multi-prompt conversational continuation format

Multi-prompt training on a conversational continuation-format dataset.

> **Note:** the source notes reused `salm_gemma4_v2_3` for this recipe. A dedicated config name (e.g. `salm_gemma4_v3_1`) is recommended — update the `--config-name` below once the file exists.

```bash
torchrun --nproc-per-node=8 /home/salm/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/salm/conf \
    --config-name=salm_gemma4_v3_1
```

## Variant summary

| Variant | Data format | Prompting | Prep script |
|---|---|---|---|
| v1 | NeMo tarred | Single-prompt | — |
| v2.1 | NeMo manifest | Multi-prompt | `nemo_add_context.py` |
| v2.2 | NeMo tarred | Multi-prompt | — |
| v2.3 | Conversational | Multi-prompt | `nemo_to_multiprompt_manifest.py` |
| v3.1 | Conversational, continuation | Multi-prompt | `nemo_to_continuation_multiprompt_manifest.py` |

## Notes

- All paths assume the `salm` user layout (`/home/salm/NeMo`, `/home/salm/conf`). Adjust if your environment differs.
- `--nproc-per-node=8` assumes a single node with 8 GPUs. For multi-node runs, set `--nnodes`, `--node-rank`, and `--master-addr`/`--master-port` accordingly.
- The original notes reused `salm_gemma4_v2_3` for both v2.3 and v3.1 — confirm a dedicated `salm_gemma4_v3_1` config exists before launching the continuation run.