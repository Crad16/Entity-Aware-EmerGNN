#!/usr/bin/env python3
"""Generate short descriptions of non-drug HetioNet entities with a local LLM served by ollama.

Genes are grounded with their NCBI full name and RefSeq summary. A description that names a drug
is regenerated (up to --retries times). Output is keyed by HetioNet id, so both datasets share it.

    python entity_text/gen_descriptions.py --sample 50 --out entity_text/sample50.jsonl
    python entity_text/gen_descriptions.py --host http://localhost:11434 --workers 8
    python entity_text/gen_descriptions.py --check entity_text/descriptions_v3.jsonl
"""
import argparse
import collections
import concurrent.futures
import csv
import gzip
import json
import os
import random
import re
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_VERSION = 'v3'
MODEL = 'qwen3:8b'

SYSTEM = ('You are a biomedical curator writing short, factual descriptions of knowledge-graph '
          'entities for a machine-learning model.')
RULES = """Describe this entity in 1-2 sentences (at most 45 words): what it is and its main biological or clinical role.
Rules:
1. Never name any specific drug, medication, chemical compound, or brand. Refer to drugs only generically (e.g., "many drugs").
2. Do not name drug classes either (e.g., no "statins" or "antidepressants").
3. Base the description only on the information given above. If no NCBI summary is given, or it only lists predicted functions, say the function is poorly characterized instead of guessing.
4. Start directly with what it does or is; do not open with a generic phrase such as "<name> is a protein-coding gene that encodes".
5. Output only the description."""
RETRY = 'Your previous answer named "{}". Rewrite it without naming any specific drug, compound, or drug class.'

# entities every sample should contain: drug-metabolism genes, PD targets, generic genes
MUST_INCLUDE = ['Gene::1565', 'Gene::1576', 'Gene::1559', 'Gene::1544', 'Gene::1557',  # CYP2D6/3A4/2C9/1A2/2C19
                'Gene::5243', 'Gene::10599', 'Gene::54658',                              # ABCB1, SLCO1B1, UGT1A1
                'Gene::3757', 'Gene::1813', 'Gene::7157', 'Gene::1956']                  # KCNH2, DRD2, TP53, EGFR
SAMPLE_PLAN = {'Gene': 13, 'Side Effect': 5, 'Disease': 3, 'Symptom': 3, 'Anatomy': 3, 'Pathway': 3,
               'Molecular Function': 3, 'Cellular Component': 2, 'Pharmacologic Class': 3}


def load_entities():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'entity_text/twosides_entities.tsv')), delimiter='\t'))
    return {r['hetionet_id']: r for r in rows if r['is_drug'] == '0'}


def load_ncbi():
    """NCBI gene_info rows plus RefSeq/Alliance summaries (entity_text/sources)."""
    path = os.path.join(ROOT, 'entity_text/sources/Homo_sapiens.gene_info.gz')
    with gzip.open(path, 'rt') as f:
        header = f.readline().lstrip('#').rstrip('\n').split('\t')
        rows = (dict(zip(header, line.rstrip('\n').split('\t'))) for line in f)
        info = {r['GeneID']: r for r in rows}
    summaries = json.load(gzip.open(os.path.join(ROOT, 'entity_text/sources/ncbi_gene_summaries.json.gz'), 'rt'))
    for gid, text in summaries.items():
        if gid in info:
            info[gid]['summary'] = re.sub(r'\s*\[provided by[^\]]*\]\.?', '', text).strip()
    return info


def kg_degree():
    deg = collections.Counter()
    for split in ['train', 'valid', 'test']:
        for line in open(os.path.join(ROOT, f'TWOSIDES/data/S1_1/{split}_KG.txt')):
            h, t, _ = line.split()
            deg[int(h)] += 1
            deg[int(t)] += 1
    return deg


def pick_sample(ents, n, seed):
    rng = random.Random(seed)
    chosen = [h for h in MUST_INCLUDE if h in ents]
    deg = kg_degree()
    by_kind = collections.defaultdict(list)
    for h, r in sorted(ents.items()):
        if h not in chosen:
            by_kind[r['kind']].append(h)
    for kind, k in SAMPLE_PLAN.items():
        pool = by_kind[kind]
        if kind == 'Gene':
            # half from the low-degree quartile so obscure genes are represented
            low = sorted(pool, key=lambda h: deg[int(ents[h]['ent_id'])])[:len(pool) // 4]
            picked = rng.sample(low, k // 2)
            chosen += picked + rng.sample([h for h in pool if h not in picked], k - k // 2)
        else:
            chosen += rng.sample(pool, k)
    return chosen[:n]


def build_prompt(hid, ent, ncbi):
    lines = [f'Entity type: {ent["kind"]}', f'Name: {ent["name"]}']
    if ent['kind'] == 'Gene':
        g = ncbi.get(hid.split('::')[1])
        if g:
            lines.append(f'Official full name (NCBI): {g["description"]}')
            lines.append(f'Gene type: {g["type_of_gene"]}')
            if g.get('summary'):
                lines.append(f'NCBI summary: {g["summary"][:700]}')
    return '\n'.join(lines) + '\n\n' + RULES


def chat(messages, host):
    body = json.dumps({'model': MODEL, 'think': False, 'stream': False, 'messages': messages,
                       'options': {'temperature': 0, 'seed': 0, 'num_predict': 150}}).encode()
    req = urllib.request.Request(f'{host}/api/chat', data=body)
    out = json.load(urllib.request.urlopen(req, timeout=300))['message']['content']
    return re.sub(r'<think>.*?</think>', '', out, flags=re.S).strip()


class LeakChecker:
    """Flags descriptions that contain a known drug name (whole-word, case-insensitive).

    Names that already occur in the entity's own name or NCBI full name (dopamine in
    "dopamine receptor D2", zinc in "zinc finger protein") are endogenous context, not leaks.
    """

    def __init__(self):
        names = json.load(open(os.path.join(ROOT, 'entity_text/drug_name_dictionary.json')))
        self.size = len(names)
        alts = sorted((re.escape(n) for n in names), key=len, reverse=True)
        # biologics often lack PubChem names; catch antibody (-mab) and fusion protein (-cept) names
        alts += [r'(?!(?:con|inter|per|pre)cept\b)\w{3,}(?:mab|cept)']
        self.pattern = re.compile(r'\b(' + '|'.join(alts) + r')\b', re.I)

    def find(self, text, grounding=''):
        allowed = {m.group(1).lower() for m in self.pattern.finditer(grounding)}
        return sorted({m.group(1).lower() for m in self.pattern.finditer(text)} - allowed)


def describe(hid, ent, ncbi, checker, args):
    prompt = build_prompt(hid, ent, ncbi)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    attempts = [chat(messages, args.host)]
    while checker.find(attempts[-1], prompt) and len(attempts) <= args.retries:
        messages += [{'role': 'assistant', 'content': attempts[-1]},
                     {'role': 'user', 'content': RETRY.format(', '.join(checker.find(attempts[-1], prompt)))}]
        attempts.append(chat(messages, args.host))
    return {'hetionet_id': hid, 'ent_id_twosides': int(ent['ent_id']), 'kind': ent['kind'],
            'name': ent['name'], 'prompt': prompt, 'description': attempts[-1],
            'attempts': attempts, 'leaked_names': checker.find(attempts[-1], prompt),
            'model': MODEL, 'prompt_version': PROMPT_VERSION}


def generate(args):
    ents, ncbi, checker = load_entities(), load_ncbi(), LeakChecker()
    done = {}
    if os.path.exists(args.out):
        done = {json.loads(l)['hetionet_id'] for l in open(args.out)}
    targets = pick_sample(ents, args.sample, args.seed) if args.sample else sorted(ents)
    todo = [t for t in targets if t not in done]
    # records are appended in completion order; the file is keyed by hetionet_id, so order does not matter
    with open(args.out, 'a') as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(describe, hid, ents[hid], ncbi, checker, args) for hid in todo]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{len(done) + i + 1}/{len(targets)}] {rec["kind"]:18s} {rec["name"][:30]:30s} '
                  f'retries={len(rec["attempts"]) - 1} leak={rec["leaked_names"]}', flush=True)


def check(path):
    checker = LeakChecker()
    recs = [json.loads(l) for l in open(path)]
    print(f'dictionary: {checker.size} drug names')
    for r in recs:
        first, final = checker.find(r['attempts'][0], r['prompt']), checker.find(r['description'], r['prompt'])
        if first or final:
            print(f'{r["name"][:30]:30s} first attempt: {first}  final: {final}')
    print(f'{sum(bool(checker.find(r["description"], r["prompt"])) for r in recs)}/{len(recs)} final descriptions still name a drug')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=0, help='0 = all non-drug entities')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--out', default=os.path.join(ROOT, f'entity_text/descriptions_{PROMPT_VERSION}.jsonl'))
    p.add_argument('--check', metavar='JSONL', help='only re-run the leak check on an existing file')
    p.add_argument('--host', default='http://localhost:11434', help='ollama server URL')
    p.add_argument('--workers', type=int, default=1, help='concurrent requests (match OLLAMA_NUM_PARALLEL)')
    a = p.parse_args()
    check(a.check) if a.check else generate(a)
