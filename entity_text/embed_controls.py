#!/usr/bin/env python3
"""Control keys for the description-content test, DrugBank only.

    llmperm  the llm keys, reassigned among non-drug entities of the same kind (Gene<->Gene, ...)
             by one fixed cyclic shift of a random order (seed 0), so no entity keeps its own key
             and the per-kind key distribution is unchanged
    desc     SapBERT CLS of the description alone: no "<name>. " prefix, and every mention of the
             entity name inside the description replaced by a kind placeholder ("this gene")

Writes entity_text/emb/drugbank_{llmperm,desc}.npy (+ drugbank_llmperm_map.npy: source ent_id per row).

    python entity_text/embed_controls.py
"""
import collections
import csv
import gzip
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from embed import MODEL, OUT, ROOT, encode

PLACEHOLDER = {'Gene': 'this gene', 'Side Effect': 'this side effect', 'Molecular Function': 'this molecular function',
               'Pathway': 'this pathway', 'Cellular Component': 'this cellular component', 'Symptom': 'this symptom',
               'Anatomy': 'this anatomical structure', 'Pharmacologic Class': 'this pharmacologic class',
               'Disease': 'this disease'}


def strip_name(name, kind, text):
    ph = PLACEHOLDER[kind]
    pat = re.compile(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', flags=re.I)

    def sub(m):
        start = m.start() == 0 or text[:m.start()].rstrip().endswith(('.', '!', '?'))
        return ph.capitalize() if start else ph
    return pat.sub(sub, text)


def main():
    ents = [r for r in csv.DictReader(open(os.path.join(ROOT, 'entity_text/drugbank_entities.tsv')), delimiter='\t')
            if r['is_drug'] == '0']
    mask = np.load(os.path.join(OUT, 'drugbank_mask.npy'))
    assert mask.sum() == len(ents) and all(mask[int(r['ent_id'])] for r in ents)

    # --- llmperm
    llm = np.load(os.path.join(OUT, 'drugbank_llm.npy'))
    rng = np.random.RandomState(0)
    by_kind = collections.defaultdict(list)
    for r in ents:
        by_kind[r['kind']].append(int(r['ent_id']))
    src = np.arange(len(llm))
    for kind, ids in sorted(by_kind.items()):
        order = np.array(ids)[rng.permutation(len(ids))]
        src[order] = np.roll(order, 1)                          # each entity takes the key of its predecessor
    on = np.where(mask)[0]
    assert (src[on] != on).all() and sorted(src[on]) == sorted(on) and (src[~mask] == np.where(~mask)[0]).all()
    np.save(os.path.join(OUT, 'drugbank_llmperm.npy'), llm[src])
    np.save(os.path.join(OUT, 'drugbank_llmperm_map.npy'), src)
    cos = (llm[on] * llm[src[on]]).sum(1)
    print(f'llmperm: {len(on)} keys moved within {len(by_kind)} kinds; cosine(own, assigned) mean {cos.mean():.3f}')

    # --- desc
    path = os.path.join(ROOT, 'entity_text/descriptions_v3.jsonl')
    recs = {r['hetionet_id']: r for r in map(json.loads, open(path) if os.path.exists(path) else gzip.open(path + '.gz', 'rt'))}
    texts, left = [], 0
    for r in ents:
        rec = recs[r['hetionet_id']]
        t = strip_name(rec['name'], r['kind'], rec['description'])
        left += rec['name'].lower() in t.lower()
        texts.append(t)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    vec = encode(texts, tokenizer, model, max_length=128)
    arr = np.zeros_like(llm)
    idx = [int(r['ent_id']) for r in ents]
    arr[idx] = vec
    np.save(os.path.join(OUT, 'drugbank_desc.npy'), arr)
    cos = (llm[idx] * vec).sum(1)
    print(f'desc: {len(texts)} texts on {device}; name string still present in {left}; cosine(llm, desc) mean {cos.mean():.3f}')
    for name in ['CYP3A4', 'ABCB1', 'Nausea']:
        for r, t in zip(ents, texts):
            if r['name'] == name:
                print(f'  {name}: {t[:150]}')
                break


if __name__ == '__main__':
    main()
