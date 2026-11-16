# Entity-Aware EmerGNN

DS535 (Fall 2026) team project. We extend [EmerGNN](https://github.com/LARS-research/EmerGNN)
(Zhang et al., *Nature Computational Science* 2023) for emerging-drug DDI prediction.

EmerGNN weighs knowledge-graph edges by relation type only, so all edges of one type (for example
every Compound–binds–Gene edge) get the same weight for a drug pair. We add an entity gate that
rescales messages leaving each non-drug entity depending on the drug pair. The entity side of the
gate comes from a text description of the entity (genes, diseases, side effects, ...) written by a
local LLM and encoded with SapBERT. Drugs never receive text.

## Changes to EmerGNN

- `entity_gate.py`: the gate, `g = 1 + 0.5 tanh(q(h_uv - mu) · W k_e / sqrt(d))` (v2). The query
  starts at zero, so training starts from the original model. `--gate_version v1` keeps the first
  variant (`2 sigmoid(...)`) for the v1 checkpoints.
- `DrugBank/`, `TWOSIDES/` (`models.py`, `base_model.py`, `evaluate.py`):
  - `--entity_feat {none,id,random,name,llm}`, `--gate_version`, `--gate_dim`, `--gate_lr`
  - `--seed` is now used (the original script always ran seed 0)
  - `--shuffle_order` (DrugBank): shuffle training pairs every epoch. `train_ddi.txt` is sorted by
    interaction type and the original loop never shuffles it
  - `--attn_mode {learned,ones,shuffle}`: relation-attention ablations
  - result and checkpoint files are named per configuration

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install torchdrug==0.2.1 --no-deps
pip install -r requirements.txt
bash scripts/patch_torchdrug.sh   # torchdrug 0.2.1 headers vs torch 2.x
```

The sparse kernel is compiled on first use and needs `nvcc` (`CUDA_HOME`, see `scripts/env.sh`).

## Data

Download the EmerGNN splits from Zenodo ([10.5281/zenodo.10016715](https://doi.org/10.5281/zenodo.10016715))
and unzip `data-DB.zip` into `DrugBank/` and `data-TS.zip` into `TWOSIDES/`, so that
`DrugBank/data/S1_1/train_ddi.txt` exists.

Entity text (already in the repository):

| file | content |
|---|---|
| `entity_text/{drugbank,twosides}_entities.tsv` | entity id → HetioNet id, name, kind |
| `entity_text/descriptions_v3.jsonl.gz` | descriptions of the 32,227 non-drug entities (qwen3:8b) |
| `entity_text/drug_name_dictionary.json` | drug names used to reject descriptions that mention drugs |
| `entity_text/sources/` | HetioNet v1.0 nodes, NCBI gene info and summaries (fetched 2026-10-02) |

To rebuild them: `build_entity_tables.py`, `fetch_ncbi.py`, `build_drug_names.py`, then
`gen_descriptions.py` (needs an ollama server with `qwen3:8b`). The vectors used by the gate are
not stored; create them with

```bash
python entity_text/embed.py      # entity_text/emb/<dataset>_{name,llm,random}.npy, ~1 min on a GPU
```

## Running

```bash
cd DrugBank
# original EmerGNN (shuffled training order)
python -W ignore evaluate.py --dataset=S1_1 --n_epoch=20 --epoch_per_test=2 --shuffle_order
# with the entity gate
python -W ignore evaluate.py --dataset=S1_1 --n_epoch=20 --epoch_per_test=2 --shuffle_order \
    --entity_feat=llm --gate_version=v2 --gate_lr=0.003 --save_model
```

Checks: `python ../test_entity_gate.py` (from `DrugBank/` or `TWOSIDES/`).

## Analyses

Run from `DrugBank/`; outputs go to `results/`.

| script | question |
|---|---|
| `analysis/analyze_gates.py` | are CYP450 gates higher on metabolism-type pairs than on other pairs? |
| `analysis/explain_shared_cyp.py` | among genes bound by both drugs, are the CYPs ranked first? |
| `analysis/rare_entity_probe.py` | does behaviour learned on frequent CYPs carry over to rare ones? |
| `analysis/name_vs_function.py` | are drug transporters grouped by function or by name? |
| `analysis/summarize.py` | best-validation epoch of a run log |
| `analysis/relation_ties.py` | how many paths share an ordered relation sequence (attention ties)? |
| `analysis/gate_interventions.py`, `analyze_interventions.py` | gate off / mean gate / query replacement / key permutation on a test sample |
| `analysis/stratify.py`, `kg_calibration.py` | results by mechanism class, shared mediator and KG coverage; calibration |
| `analysis/kg_variants.py`, `kg_ablation.py`, `analyze_kg_ablation.py` | test-time KG deletions: CYP edges, equal-count controls, 50/90/100% of an emerging drug's edges |
| `analysis/gate_degree.py` | within-pair CYP-ranking AUC, including candidates with no degree advantage |

The intervention analyses take `--dataset`, `--train_seed` and `--keys`; `analysis/REPRODUCE.md`
lists the order of the steps, the seeds, and where each number of the progress report comes from.

## Results

`results/curves/` has the validation/test curve of every run (seed 0) and `results/*.json` the
analysis outputs. Trained DrugBank models are in `DrugBank/saved_models/`.

DrugBank S1_1, macro-F1 on test, mean over epochs 10–20 (single seed):

| model | F1 |
|---|---|
| EmerGNN | 0.617 |
| + gate v2, ID keys | 0.628 |
| + gate v2, random keys | 0.630 |
| + gate v2, name keys | 0.602 |
| + gate v2, LLM keys | 0.610 |

## Layout

```
DrugBank/, TWOSIDES/   EmerGNN code per dataset (modified), saved_models/
entity_gate.py         entity gate
entity_text/           entity tables, descriptions, embedding scripts
analysis/              gate analyses
scripts/               environment, torchdrug patch, GPU/tmux launcher, curve export
results/               curves and analysis outputs
```

## Acknowledgements

Built on EmerGNN by Yongqi Zhang et al. (MIT license, see `LICENSE` and `CITATION.cff`).
Data: DrugBank, TWOSIDES and HetioNet via the EmerGNN Zenodo release.
