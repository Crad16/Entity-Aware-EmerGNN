# Intervention analyses: how each reported number was produced

All commands run from `DrugBank/`. Every script takes

    --dataset S1_1        split (S1_1, S1_12, S1_123, ...)
    --train_seed 0        the seed<k> in the checkpoint names
    --keys llm id random name    gate key types to analyse (S1_1 seed 0 has all four; `llm@last`
                                 selects the last-epoch checkpoint instead of the validation-best one)

Large files (probabilities, `.npz`) go to `runs/<tag>/` (git-ignored), small tables to
`results/<tag>/`, where `<tag>` is the split, or `<split>_seed<k>` for a training seed other than 0.
Checkpoints must be named
`saved_models/<split>_learned_shuf[_gate-<key>-v2]_seed<k>_<best|last>.pt` (what `evaluate.py
--save_model` writes); a checkpoint whose split or seed differs from the command line is refused.

## Order of the steps (one split)

```bash
D=S1_12; K="llm name"
# 0. keys (once): python ../entity_text/embed.py   ->  entity_text/emb/drugbank_{name,llm}.npy

# 1. path ties (CPU, ~15 s; no model)
python ../analysis/relation_ties.py --dataset $D

# 2. gate interventions (GPU): draws the test sample and saves probabilities per condition
python ../analysis/gate_interventions.py --dataset $D \
    --ckpt saved_models/${D}_learned_shuf_gate-llm-v2_seed0_best.pt saved_models/${D}_learned_shuf_gate-name-v2_seed0_best.pt
python ../analysis/analyze_interventions.py --dataset $D --keys $K      # CPU

# 3. strata and calibration (CPU; read the probabilities of step 2)
python ../analysis/stratify.py --dataset $D --keys $K
python ../analysis/kg_calibration.py --dataset $D --keys $K

# 4. KG deletions (GPU), gated models and the ungated baseline, same sample as step 2
python ../analysis/kg_variants.py --dataset $D                          # prints what each variant removes
python ../analysis/kg_ablation.py --dataset $D --variants cyp hub rand0 rand1 new50 new90 new100 \
    --ckpt saved_models/${D}_learned_shuf_gate-llm-v2_seed0_best.pt saved_models/${D}_learned_shuf_gate-name-v2_seed0_best.pt
python ../analysis/kg_ablation.py --dataset $D --variants full cyp hub rand0 rand1 new50 new90 new100 \
    --ckpt saved_models/${D}_learned_shuf_seed0_best.pt
python ../analysis/analyze_kg_ablation.py --dataset $D --keys $K        # CPU

# 5. shared-CYP ranking (CPU, reads only the gate parameters of the checkpoints)
python ../analysis/gate_degree.py --dataset $D --keys $K --ckpts best

# 6. calibration with a validation temperature (GPU for the predictions, then CPU)
python ../analysis/valid_probs.py --dataset $D --ckpt saved_models/${D}_learned_shuf_seed0_best.pt \
    saved_models/${D}_learned_shuf_gate-llm-v2_seed0_best.pt saved_models/${D}_learned_shuf_gate-name-v2_seed0_best.pt
python ../analysis/valid_calibration.py --dataset $D --train_seeds 0 --keys $K

# 7. deletion of gate-ranked entities (GPU, ~1 h per seed on an RTX 3080), then the tables (CPU)
python ../analysis/gate_rank_deletion.py --dataset $D \
    --ckpt saved_models/${D}_learned_shuf_gate-llm-v2_seed0_best.pt saved_models/${D}_learned_shuf_gate-name-v2_seed0_best.pt \
    --baseline saved_models/${D}_learned_shuf_seed0_best.pt
python ../analysis/analyze_gate_rank_deletion.py --dataset $D --train_seeds 0 --keys $K

# 8. gated models against the baseline (CPU; needs steps 2 and 4)
python ../analysis/gate_vs_baseline.py --dataset $D --train_seeds 0 --keys $K
```

For another training seed add `--train_seed <k>` to steps 2-5 and to the GPU commands of steps 6-7,
with the `seed<k>` checkpoints. S1_1 was run for seeds 0, 1, 2; steps 6-8 and `compare_seeds.py`
then take `--train_seeds 0 1 2` and write to `results/S1_1/`.

Notes
- Step 2 must come before steps 3 and 4: they reuse its sample (`rows`, `weight` inside every
  `runs/<tag>/gate_interventions/*_probs.npz`). For S1_1 the pair list is also in
  `results/S1_1/test_sample.npz` (row indices into `data/S1_1/test_ddi.txt`, and weights). A sample
  is specific to its split.
- An ungated checkpoint has no gate, so it only goes through `kg_ablation.py`; include the variant
  `full` to get its full-KG predictions (`runs/<tag>/kg_ablation/baseline_best__<variant>.npz`, only
  the `original` condition). `analyze_kg_ablation.py` adds them as the model `baseline` when they are
  present.
- `off` = the gated model with every gate set to 1 (same computation as EmerGNN, weights trained
  together with the gate). `emergnn_x4/off` in `kg_ablation.json` is the mean of `off` over the keys
  given by `--keys` (four on S1_1).
- With `--keys llm name`, `stratify.py` averages `off_acc` / `off_conf` over those two models, which
  is how the report's stratum numbers were computed.
- Sample accuracy matches full-test accuracy within 0.6 points; sample macro-F1 does not (rare
  types), so only accuracy is reported from the sample.
- Runtime on one H100: `kg_ablation.py` about 26 min for 8 variants of the ungated model (twice
  that per gated model, which runs `original` and `off`). On an RTX 3080: about 14 min and 25 min.

## Seeds

| what | value |
|---|---|
| test sample | `RandomState(0)` (`--seed 0`), 8,192 requested, drawn per DDI type in proportion, at least one pair per type (8,195 pairs on S1_1); weight = type size / pairs drawn |
| mean gate | 4,096 training pairs, same generator, drawn after the sample |
| query replacement donors | `RandomState(1000*seed + 10*rep + within_type)`, `rep` = 0, 1, 2; `within_type` 1 = same DDI type, 0 = any pair |
| key permutation | `RandomState(2000*seed + rep)`, `rep` = 0, 1, 2; within entity kind (Gene with Gene, ...) |
| bootstrap | `RandomState(0)`, 1,000 resamples of emerging drugs (clusters), percentile 95% interval |
| KG variants | `rand<k>`: `RandomState(1000*0 + k + 1)`; `new50/90/100`: `RandomState(0)`, nested |
| gene bootstrap of the CYP regression coefficient (`gate_degree.py`) | `RandomState(0)`, 1,000 resamples |

## Reported item -> file and field (S1_1, seed 0, validation-best checkpoints)

Checkpoints: `S1_1_learned_shuf_gate-llm-v2_seed0_best.pt` (epoch 16) and
`S1_1_learned_shuf_gate-name-v2_seed0_best.pt` (epoch 20). The id and random keys are in the result
files but not in the report. Seeds 1 and 2 have the same files under `results/S1_1_seed<k>/`.

| item | script | file under `results/S1_1/` | field |
|---|---|---|---|
| macro-F1 epoch 10-20 | `analysis/summarize.py --curves <training log>` | `../curves/*.csv` | test F1 averaged over the evaluations at epochs 10, 12, ..., 20 (not computed by a script) |
| tie rates 88.9% / 69.3% | `relation_ties.py` | `relation_ties.json` | `2hop`, `3hop` |
| group size 31 (side effect) and 2 (binding) | `relation_ties.py` | `relation_ties.json` | `top_2hop_sequences/*/k_median`, `binds_binds` |
| tie size 8.0 / 24.6 / 45.0 by degree band | `stratify.py` | `stratify.json` | `kg=low|mid|high / tie_size_median` (c²-weighted: Σc²/Σc over relation sequences) |
| metabolism-label rate, PD and other-PK, with / without shared CYP | `analyze_kg_ablation.py` | `kg_ablation.json` | `f4/<key>/original/pull/<class>&shared_cyp|no_shared_cyp/full` |
| accuracy 48.1%, confidence 88.0% on shared-CYP PD pairs | `stratify.py --keys llm name` | `stratify.json` | `pd & cyp / off_acc, off_conf` (the stored file averages four keys: .508 / .887) |
| accuracy after CYP removal and controls | `kg_ablation.py`, `analyze_kg_ablation.py` | `kg_ablation.json` | `accuracy/<key>/original` (`full`, `cyp`, `hub`, `rand0`, `rand1`) |
| change in log p for metabolism pairs | same | `kg_ablation.json` | `f4/<key>/<original|off>/dlogp_cyp/metabolism` |
| metabolism share .42 -> .19, precision .73 -> .76 | same | `kg_ablation.json` | `f4/<key>/original/shared_cyp_metab_pred/<full|cyp>/<rate|precision>` |
| key permutation among shared-CYP pairs (n = 1,553 / 1,113) | `analyze_interventions.py` | `gate_interventions/followup.json` | `<key>/<metabolism|non_pk>/key_perm-original_with_shared_cyp` |
| query replacement, KL difference -.013 / -.029 | `analyze_interventions.py` | `gate_interventions/followup.json` | `<key>/metabolism/kl_type-all` |
| aggregate accuracy under each intervention | `gate_interventions.py` | `gate_interventions/<key>-v2_best.json` | `summary/<condition>/acc` |
| CYP-ranking AUC and contrast | `gate_degree.py` | `gate_degree.json` | `<key>-v2 [best]/e2/<metabolism|non_pk|gap>/auc`; rows `EmerGNN (ties)`, `degree heuristic` |
| no degree advantage subset | `gate_degree.py` | `gate_degree.json` | `.../e2/<group>/adverse_auc` |
| gate vs degree correlation <= .09 | `gate_degree.py` | `gate_degree.json` | `<key>-v2 [best]/e4/*_vs_*deg*`, `within_pair_rho_mean` |
| accuracy after removing 90% / 100% of an emerging drug's edges | `kg_ablation.py`, `analyze_kg_ablation.py` | `kg_ablation.json` | `accuracy/<key>/original` (`new90`, `new100`) |
| heuristic .529 | `analyze_kg_ablation.py` | `kg_ablation.json` | `m1_reference/existing_drug_top_type` |
| natural degree variation, temperature | `kg_calibration.py` | `kg_calibration.json` | `models/<key>/<cond>/raw_drug_degree`, `temp_*`, `T` |
| accuracy with the ungated baseline in the same table | `analyze_kg_ablation.py --keys llm name --out ...` | `kg_ablation_llm_name_baseline.json` | `accuracy/baseline/original`, `f4/baseline/original` |
| gate minus baseline, per seed and pooled | `gate_vs_baseline.py` | `gate_vs_baseline.json` | `rows/<variant>|<stratum>|<key>-baseline|<seed or pooled>` |
| calibration with the validation temperature | `valid_calibration.py` | `valid_calibration.json` | gap and ECE per stratum: raw, validation T, oracle T |
| deletion of gate-ranked entities | `analyze_gate_rank_deletion.py` | `gate_rank_deletion.json` | `rows/<key>|<condition>|<measure>|<set or contrast>|<class>|<seed or pooled>` |
| control keys under KG deletion | `control_keys_sparse_kg.py` | `control_keys_sparse_kg.json` | `accuracy`, `diff` |

Type classes (metabolism, other PK, PD, unspecified) for all 86 types: `results/S1_1/ddi_type_classes.tsv`
(written by `stratify.py`; the class column is the same for every split, the pair counts are not).

## Baseline

The ungated baselines are `DrugBank/saved_models/S1_1_learned_shuf_seed<k>_best.pt` (20 epochs; seed 0:
best epoch 14, test accuracy 0.717). They only go through step 4 (with the variant `full`), step 6
and, as `--baseline`, step 7.
