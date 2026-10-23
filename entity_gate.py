"""Entity-aware gate shared by the TWOSIDES and DrugBank models.

Messages leaving a non-drug entity e are scaled by a gate g_uv(e) that depends on the drug pair,
so an edge weight becomes alpha_r(u, v) * g_uv(source entity).

    v1: g = 2 * sigmoid(q_l(h_uv) . W k_e)                      all layers, range (0, 2)
    v2: g = 1 + 0.5 * tanh(q_l(h_uv - mu) . W k_e / sqrt(d))    layers >= 1, range (0.5, 1.5)

h_uv is the pair embedding, mu the mean pair embedding over all drugs, k_e the entity vector
(--entity_feat: learned 'id', or a frozen 'random' / 'name' / 'llm' vector). The query starts at
zero, so the model equals EmerGNN at initialization. Drug entities have no key and keep g = 1.
"""
import csv
import os
import re

import numpy as np
import torch
import torch.nn as nn


class EntityGate(nn.Module):
    def __init__(self, feat, emb_prefix, n_ent, n_dim, n_layer, gate_dim, version='v1'):
        super().__init__()
        assert version in ('v1', 'v2'), version
        self.feat, self.version, self.n_layer = feat, version, n_layer
        self.lo, self.hi = (0.0, 2.0) if version == 'v1' else (0.5, 1.5)
        self.scale = 1.0 if version == 'v1' else gate_dim ** 0.5
        mask = torch.from_numpy(np.load(f'{emb_prefix}_mask.npy'))
        assert len(mask) == n_ent, (len(mask), n_ent)
        self.register_buffer('mask', mask.float().unsqueeze(1), persistent=False)
        if feat == 'id':
            self.key = nn.Embedding(n_ent, gate_dim)
            nn.init.normal_(self.key.weight, std=gate_dim ** -0.5)
        else:
            emb = torch.from_numpy(np.load(f'{emb_prefix}_{feat}.npy'))
            self.register_buffer('emb', emb, persistent=False)
            self.proj = nn.Linear(emb.size(1), gate_dim, bias=False)
        self.layers = list(range(n_layer)) if version == 'v1' else list(range(1, n_layer))
        self.query = nn.ModuleList([nn.Linear(2 * n_dim, gate_dim, bias=version == 'v1') for _ in self.layers])
        for q in self.query:
            nn.init.zeros_(q.weight)
            if q.bias is not None:
                nn.init.zeros_(q.bias)
        for name, idx in entity_groups(emb_prefix).items():
            self.register_buffer(f'{name}_idx', idx, persistent=False)

    def keys(self):
        k = self.key.weight if self.feat == 'id' else self.proj(self.emb)
        return k * self.mask                                     # (n_ent, gate_dim), drug rows zero

    def forward(self, ht_embed, keys=None, center=None):
        """Gates per layer: a list of (n_ent, batch) tensors, None where a layer has no gate."""
        keys = self.keys() if keys is None else keys
        if self.version == 'v2':
            ht_embed = ht_embed - center
        gates = [None] * self.n_layer
        for l, q in zip(self.layers, self.query):
            logits = keys @ q(ht_embed).t() / self.scale
            gates[l] = 2 * torch.sigmoid(logits) if self.version == 'v1' else 1 + 0.5 * torch.tanh(logits)
        return gates

    @torch.no_grad()
    def summary(self, ht_embed, center=None):
        """One-line gate statistics for logging (non-drug entities, gated layers)."""
        gates = torch.stack([g for g in self.forward(ht_embed, center=center) if g is not None])  # (L', n_ent, batch)
        on = self.mask.squeeze(1).bool()
        g = gates[:, on]
        cyp = gates[:, self.cyp_idx].mean().item()
        gene = gates[:, self.gene_idx].mean().item()
        margin = 0.025 * (self.hi - self.lo)
        saturated = ((g < self.lo + margin) | (g > self.hi - margin)).float().mean().item()
        # share of gate variance that varies with the drug pair (rest is a fixed per-entity offset)
        total = g.var(dim=(1, 2), unbiased=False)
        between_entity = g.mean(2).var(dim=1, unbiased=False)
        pair_share = (1 - between_entity / total.clamp_min(1e-12)).mean().item()
        return (f'gate: mean {g.mean().item():.4f} std {g.std().item():.4f} '
                f'min {g.min().item():.3f} max {g.max().item():.3f} cyp {cyp:.4f} other_gene {gene:.4f} '
                f'saturated {saturated:.3f} pair_share {pair_share:.3f}')


def entity_groups(emb_prefix):
    """Index tensors of CYP450 genes and other genes, read from entity_text/<dataset>_entities.tsv."""
    emb_dir = os.path.dirname(emb_prefix)
    tsv = os.path.join(os.path.dirname(emb_dir), os.path.basename(emb_prefix) + '_entities.tsv')
    cyp, gene = [], []
    for r in csv.DictReader(open(tsv), delimiter='\t'):
        if r['kind'] == 'Gene':
            (cyp if re.match(r'CYP\d', r['name']) else gene).append(int(r['ent_id']))
    return {'cyp': torch.tensor(cyp), 'gene': torch.tensor(gene)}


def apply_gate(hiddens, gate, n_ent, batch):
    """Scale hidden states (n_ent, batch * dim) of each source entity by its gate (n_ent, batch)."""
    return (hiddens.view(n_ent, batch, -1) * gate.unsqueeze(2)).view(n_ent, -1)
