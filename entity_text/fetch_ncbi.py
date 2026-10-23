#!/usr/bin/env python3
"""Download the NCBI gene information used to ground gene descriptions.

    entity_text/sources/Homo_sapiens.gene_info.gz       official symbol / full name
    entity_text/sources/ncbi_gene_summaries.json.gz      RefSeq / Alliance summary per gene

The copies in the repository were fetched on 2026-10-02; NCBI updates both continuously.

    python entity_text/fetch_ncbi.py
"""
import csv
import gzip
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'entity_text/sources')
GENE_INFO = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'


def main():
    os.makedirs(OUT, exist_ok=True)
    urllib.request.urlretrieve(GENE_INFO, os.path.join(OUT, 'Homo_sapiens.gene_info.gz'))

    tsv = os.path.join(ROOT, 'entity_text/twosides_entities.tsv')
    genes = [r['hetionet_id'].split('::')[1] for r in csv.DictReader(open(tsv), delimiter='\t') if r['kind'] == 'Gene']
    summaries = {}
    for i in range(0, len(genes), 300):
        body = urllib.parse.urlencode({'db': 'gene', 'id': ','.join(genes[i:i + 300]), 'retmode': 'json'}).encode()
        for attempt in range(4):
            try:
                result = json.load(urllib.request.urlopen(ESUMMARY, data=body, timeout=120))['result']
                break
            except Exception:
                time.sleep(3)
        for uid in result.get('uids', []):
            summaries[uid] = result[uid].get('summary', '')
        time.sleep(0.4)  # stay under the 3 requests/s limit without an API key
    with gzip.open(os.path.join(OUT, 'ncbi_gene_summaries.json.gz'), 'wt') as f:
        json.dump(summaries, f)
    print(f'{len(summaries)}/{len(genes)} genes, {sum(map(bool, summaries.values()))} with a summary')


if __name__ == '__main__':
    main()
