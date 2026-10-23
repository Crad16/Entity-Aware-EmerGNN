#!/usr/bin/env python3
"""Map EmerGNN entity ids to HetioNet ids, names and kinds.

Writes entity_text/{twosides,drugbank}_entities.tsv. Drugs (DDI nodes and HetioNet compounds) are
marked is_drug=1 and never receive text.

    python entity_text/build_entity_tables.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_nodes():
    with open(os.path.join(ROOT, 'entity_text/sources/hetionet-v1.0-nodes.tsv')) as f:
        rows = csv.DictReader(f, delimiter='\t')
        return {r['id']: (r['name'], r['kind']) for r in rows}


def write(path, drug_rows, entity_map, nodes):
    rows = []
    for ent_id, hid, note in drug_rows:
        name = nodes[hid][0] if hid in nodes else ''
        rows.append((ent_id, hid or '', name, 'Drug(DDI node)', 1, note))
    for hid, ent_id in entity_map.items():
        name, kind = nodes[hid]
        rows.append((ent_id, hid, name, kind, int(kind == 'Compound'), ''))
    rows.sort()
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['ent_id', 'hetionet_id', 'name', 'kind', 'is_drug', 'note'])
        w.writerows(rows)
    print(f'{path}: {len(rows)} entities, {sum(1 for r in rows if not r[4])} non-drug')


def main():
    nodes = load_nodes()
    # TWOSIDES: drugs 0..603 come with a PubChem CID and, for most, a DrugBank id
    d = os.path.join(ROOT, 'TWOSIDES/data')
    drugs = json.load(open(os.path.join(d, 'id2drug.json')))
    drug_rows = [(int(k), 'Compound::' + v['db'] if v['db'] else None, v['cid']) for k, v in drugs.items()]
    write(os.path.join(ROOT, 'entity_text/twosides_entities.tsv'), drug_rows,
          json.load(open(os.path.join(d, 'entity2id.json'))), nodes)
    # DrugBank: drugs 0..1709 are DrugBank ids
    d = os.path.join(ROOT, 'DrugBank/data')
    drug_rows = [(v, 'Compound::' + k, '') for k, v in json.load(open(os.path.join(d, 'node2id.json'))).items()]
    write(os.path.join(ROOT, 'entity_text/drugbank_entities.tsv'), drug_rows,
          json.load(open(os.path.join(d, 'entity_drug.json'))), nodes)


if __name__ == '__main__':
    main()
