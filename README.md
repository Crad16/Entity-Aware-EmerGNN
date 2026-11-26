# Entity-Aware EmerGNN

DS535 (Fall 2026) team project. We extend [EmerGNN](https://github.com/LARS-research/EmerGNN)
(Zhang et al., *Nature Computational Science* 2023) for emerging-drug DDI prediction.

EmerGNN weighs knowledge-graph edges by relation type only, so all edges of one type (for example
every Compound-binds-Gene edge) get the same weight for a drug pair. We add an entity gate that
rescales messages leaving each non-drug entity depending on the drug pair. The entity side of the
gate comes from a text description of the entity (genes, diseases, side effects, ...) written by a
local LLM and encoded with SapBERT. Drugs never receive text.

The gate does not change accuracy. We use it to look at what the model relies on: on DrugBank S1 the
answer is mostly the CYP450 binding edges, in the original EmerGNN as much as in the gated models.
The main numbers are under [Results](#results).

## Changes to EmerGNN

- `entity_gate.py`: the gate, `g = 1 + 0.5 tanh(q(h_uv - mu) · W k_e / sqrt(d))` (v2). The query
  starts at zero, so training starts from the original model. `--gate_version v1` keeps the first
  variant (`2 sigmoid(...)`) for the v1 checkpoints.
- `DrugBank/`, `TWOSIDES/` (`models.py`, `base_model.py`, `evaluate.py`):
  - `--entity_feat {none,id,random,name,llm,llmperm,desc}`, `--gate_version`, `--gate_dim`, `--gate_lr`
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

The sparse kernel is compiled on first use and needs `nvcc`. `scripts/env.sh` activates the venv and
sets `CUDA_HOME` (default `/usr/local/cuda`); put machine-specific values in `scripts/env.local.sh`.

## Data

Download the EmerGNN splits from Zenodo ([10.5281/zenodo.10016715](https://doi.org/10.5281/zenodo.10016715))
and unzip `data-DB.zip` into `DrugBank/` and `data-TS.zip` into `TWOSIDES/`, so that
`DrugBank/data/S1_1/train_ddi.txt` exists.

Entity text (already in the repository):

| file | content |
|---|---|
| `entity_text/{drugbank,twosides}_entities.tsv` | entity id to HetioNet id, name, kind |
| `entity_text/descriptions_v3.jsonl.gz` | descriptions of the 32,227 non-drug entities (qwen3:8b) |
| `entity_text/drug_name_dictionary.json` | drug names used to reject descriptions that mention drugs |
| `entity_text/sources/` | HetioNet v1.0 nodes, NCBI gene info and summaries (fetched 2026-10-02) |

To rebuild them: `build_entity_tables.py`, `fetch_ncbi.py`, `build_drug_names.py`, then
`gen_descriptions.py` (needs an ollama server with `qwen3:8b`). The vectors used by the gate are
not stored; create them with

```bash
python entity_text/embed.py           # entity_text/emb/<dataset>_{name,llm,random}.npy, ~1 min on a GPU
python entity_text/embed_controls.py  # control keys: llmperm (llm keys moved to another entity of the
                                      # same kind) and desc (description with the entity name removed)
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

Run from `DrugBank/`. Small tables go to `results/<tag>/`, saved probabilities to `runs/<tag>/`
(not tracked), where `<tag>` is `S1_1` for training seed 0 and `S1_1_seed<k>` otherwise.

| script | question |
|---|---|
| `relation_ties.py` | how many paths share an ordered relation sequence (attention ties)? |
| `gate_interventions.py`, `analyze_interventions.py` | gate off / mean gate / query replacement / key permutation on a test sample |
| `stratify.py`, `kg_calibration.py` | results by mechanism class, shared mediator and KG coverage |
| `kg_variants.py`, `kg_ablation.py`, `analyze_kg_ablation.py` | test-time KG deletions: CYP edges, equal-count controls, 50/90/100% of an emerging drug's edges |
| `gate_degree.py` | within-pair CYP-ranking AUC, including candidates with no degree advantage |
| `gate_rank_deletion.py`, `analyze_gate_rank_deletion.py` | delete the entities a gate ranks high or low for each pair, with degree-matched sets, also in the ungated model |
| `valid_probs.py`, `valid_calibration.py` | calibration with a temperature fitted on the validation pairs |
| `gate_vs_baseline.py`, `compare_seeds.py` | gated models against the ungated baseline, per seed and pooled |
| `control_keys_sparse_kg.py` | KG deletions for the control keys (id, random, desc, llmperm; seed 0) |
| `summarize.py` | best-validation epoch of a run log |

Earlier exploratory scripts (v1 gate): `analyze_gates.py`, `explain_shared_cyp.py`,
`rare_entity_probe.py`, `name_vs_function.py`; their outputs are the json files directly under `results/`.

`analysis/REPRODUCE.md` has the order of the steps, the seeds, and the file and field behind each
reported number.

## Results

DrugBank S1_1, three training seeds for EmerGNN and the two text keys. Intervals are 95% bootstrap
over emerging drugs; the intervention numbers are on a type-stratified sample of 8,195 test pairs.

Test macro-F1, mean over epochs 10-20 (`results/curves/`):

| model | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| EmerGNN | 0.617 (a) | 0.593 | 0.602 | 0.597 ± 0.007 (b) |
| + gate, name keys | 0.602 | 0.582 | 0.613 | 0.599 ± 0.016 |
| + gate, LLM keys | 0.610 | 0.594 | 0.595 | 0.599 ± 0.009 |

(a) from a 40-epoch run; the seed-0 checkpoint used in the analyses is a separate 20-epoch run.
(b) seeds 1 and 2.
Single runs with other keys (seed 0): id 0.628, random 0.630, desc 0.599, llmperm 0.602.

Accuracy on the sample when KG edges are removed at test time (seed 0 / 1 / 2):

| model | full KG | no CYP edges (2,684) | no hub-gene edges (2,690) | emerging drug: 90% removed | 100% removed |
|---|---|---|---|---|---|
| EmerGNN | .714 / .705 / .704 | .662 / .644 / .645 | .710 / .689 / .698 | .451 / .426 / .485 | .310 / .303 / .378 |
| + gate, name keys | .695 / .671 / .713 | .659 / .621 / .679 | .692 / .659 / .709 | .485 / .444 / .544 | .376 / .329 / .438 |
| + gate, LLM keys | .716 / .701 / .692 | .645 / .666 / .654 | .702 / .699 / .682 | .470 / .536 / .503 | .371 / .413 / .398 |

- Paths tie often: 88.9% of 2-hop and 69.3% of 3-hop paths share their relation sequence with
  another path of the same pair (`relation_ties.json`).
- The CYP binding edges are 0.16% of the KG triples; removing them costs 3 to 7 points of accuracy
  and about one unit of log-probability on metabolism pairs, in all nine models. Equal-count
  controls cost at most 1.6 points.
- PD pairs whose drugs share a CYP are predicted as a metabolism type in 11-19% of cases (1-3%
  without a shared gene). They stay over-confident after temperature scaling on validation data
  (accuracy about .48, confidence about .69), again in all nine models (`valid_calibration.json`).
- Permuting gene keys lowers log p of the true type on metabolism pairs with a shared CYP in
  every gated model (-0.06 to -0.12).
- Deleting the entities a gate ranks high hurts metabolism pairs more than deleting the ones it
  ranks low, also with degree-matched sets, and the same sets matter equally in the ungated model
  (`gate_rank_deletion.json`).
- With 90-100% of the emerging drug's edges removed the gated models keep 4 to 6 points more
  accuracy than EmerGNN, with the gate on or switched off at test time
  (`gate_vs_baseline.json`); all models are then below the .529 obtained by
  predicting the most frequent type of the known drug.

Trained DrugBank models are in `DrugBank/saved_models/`
(`<split>_learned_shuf[_gate-<key>-v2]_seed<k>_<best|last>.pt`).

## Layout

```
DrugBank/, TWOSIDES/   EmerGNN code per dataset (modified), saved_models/
entity_gate.py         entity gate
entity_text/           entity tables, descriptions, embedding scripts
analysis/              analyses, REPRODUCE.md
scripts/               environment, torchdrug patch, GPU/tmux launcher, curve export
results/               curves (per run) and analysis outputs (per split and seed)
```

## Acknowledgements

Built on EmerGNN by Yongqi Zhang et al. (MIT license, see `LICENSE` and `CITATION.cff`).
Data: DrugBank, TWOSIDES and HetioNet via the EmerGNN Zenodo release.
