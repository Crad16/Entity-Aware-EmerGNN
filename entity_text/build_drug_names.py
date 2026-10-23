#!/usr/bin/env python3
"""Build the drug-name dictionary used to reject descriptions that mention a drug.

Sources: HetioNet compound names, PubChem titles of the TWOSIDES drugs (by CID), and PubChem
synonyms of DrugBank drugs that are not HetioNet compounds. Biologics without a PubChem entry are
caught by a -mab / -cept pattern in gen_descriptions.py instead.

    python entity_text/build_drug_names.py
"""
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUG = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def get_json(url, data=None):
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=60))
        except Exception:
            time.sleep(2)
    return None


def main():
    names = {}
    nodes = os.path.join(ROOT, 'entity_text/sources/hetionet-v1.0-nodes.tsv')
    hetionet_compounds = set()
    for line in list(open(nodes))[1:]:
        hid, name, kind = line.rstrip('\n').split('\t')
        if kind == 'Compound':
            names[name.lower()] = 'hetionet'
            hetionet_compounds.add(hid)

    drugs = json.load(open(os.path.join(ROOT, 'TWOSIDES/data/id2drug.json')))
    cids = [str(int(v['cid'].replace('CID', ''))) for v in drugs.values()]
    for i in range(0, len(cids), 100):
        data = urllib.parse.urlencode({'cid': ','.join(cids[i:i + 100])}).encode()
        res = get_json(f'{PUG}/compound/cid/property/Title/JSON', data) or {}
        for p in res.get('PropertyTable', {}).get('Properties', []):
            if p.get('Title'):
                names.setdefault(p['Title'].lower(), 'pubchem_twosides')

    node2id = json.load(open(os.path.join(ROOT, 'DrugBank/data/node2id.json')))
    for db in (d for d in node2id if 'Compound::' + d not in hetionet_compounds):
        for kind in ['compound', 'substance']:
            res = get_json(f'{PUG}/{kind}/xref/RegistryID/{db}/synonyms/JSON')
            if res:
                syn = [s for s in res['InformationList']['Information'][0].get('Synonym', [])
                       if not s.upper().startswith('DB') and len(s) >= 4]
                if syn:
                    names.setdefault(syn[0].lower(), 'pubchem_drugbank')
                    break
            time.sleep(0.22)

    names = {k: v for k, v in names.items() if len(k) >= 4}
    json.dump(names, open(os.path.join(ROOT, 'entity_text/drug_name_dictionary.json'), 'w'), indent=0)
    print(f'{len(names)} drug names')


if __name__ == '__main__':
    main()
