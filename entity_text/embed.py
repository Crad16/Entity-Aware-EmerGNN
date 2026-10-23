#!/usr/bin/env python3
"""Encode entities with SapBERT (CLS vector, unit norm); drug rows stay zero.

Variants: name ("CYP2D6"), llm ("<name>. <description>"), random (Gaussian, keyed by HetioNet id).
Writes entity_text/emb/<dataset>_{name,llm,random}.npy and <dataset>_mask.npy.

    python entity_text/embed.py
"""
import csv
import gzip
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'entity_text/emb')
MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
DATASETS = {'twosides': 'entity_text/twosides_entities.tsv', 'drugbank': 'entity_text/drugbank_entities.tsv'}


@torch.no_grad()
def encode(texts, tokenizer, model, max_length, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(model.device)
        out.append(model(**batch).last_hidden_state[:, 0].float().cpu())
    vecs = torch.cat(out).numpy()
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(ROOT, 'entity_text/descriptions_v3.jsonl')
    lines = open(path) if os.path.exists(path) else gzip.open(path + '.gz', 'rt')
    recs = {r['hetionet_id']: r for r in map(json.loads, lines)}
    hids = sorted(recs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    names = [recs[h]['name'] for h in hids]
    texts = [f'{recs[h]["name"]}. {recs[h]["description"]}' for h in hids]
    n_trunc = sum(len(tokenizer(t)['input_ids']) > 128 for t in texts)
    vec = {
        'name': encode(names, tokenizer, model, max_length=32),
        'llm': encode(texts, tokenizer, model, max_length=128),
    }
    rnd = np.random.default_rng(0).standard_normal((len(hids), 768)).astype(np.float32)
    vec['random'] = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)
    row = {h: i for i, h in enumerate(hids)}
    print(f'encoded {len(hids)} entities on {device}; descriptions truncated at 128 tokens: {n_trunc}')

    for dataset, tsv in DATASETS.items():
        ents = list(csv.DictReader(open(os.path.join(ROOT, tsv)), delimiter='\t'))
        n_ent = max(int(r['ent_id']) for r in ents) + 1
        mask = np.zeros(n_ent, dtype=bool)
        idx_ent, idx_vec = [], []
        for r in ents:
            if r['is_drug'] == '0':
                idx_ent.append(int(r['ent_id']))
                idx_vec.append(row[r['hetionet_id']])
        mask[idx_ent] = True
        np.save(os.path.join(OUT, f'{dataset}_mask.npy'), mask)
        for variant, v in vec.items():
            arr = np.zeros((n_ent, v.shape[1]), dtype=np.float32)
            arr[idx_ent] = v[idx_vec]
            np.save(os.path.join(OUT, f'{dataset}_{variant}.npy'), arr)
        print(f'{dataset}: {n_ent} entities, {mask.sum()} with vectors')

    # sanity check: nearest neighbours of a few entities in each text variant
    for probe in ['Gene::1565', 'Gene::3757', 'Side Effect::C0027497']:
        if probe not in row:
            continue
        for variant in ['name', 'llm']:
            sims = vec[variant] @ vec[variant][row[probe]]
            top = [recs[hids[j]]['name'] for j in np.argsort(-sims)[1:6]]
            print(f'  {recs[probe]["name"]:8s} [{variant:4s}] nearest: {top}')


if __name__ == '__main__':
    main()
