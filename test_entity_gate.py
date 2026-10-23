"""Sanity checks for the entity gate (v1 and v2, every key type).

At initialization all gates are 1 and the output matches the original model; after one
gradient step the non-drug gates move and the drug gates stay at 1.

    cd DrugBank && python ../test_entity_gate.py --dataset S1_1
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from load_data import DataLoader  # noqa: E402
from models import EmerGNN  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='S1_1')
p.add_argument('--batch', type=int, default=8)
cli = p.parse_args()

dataset_dir = os.path.basename(os.getcwd())
args = argparse.Namespace(task_dir='./', dataset=cli.dataset, length=3, feat='M', attn_mode='learned',
                          gate_dim=32, emb_prefix=f'../entity_text/emb/{dataset_dir.lower()}',
                          n_dim=64 if dataset_dir == 'DrugBank' else 32, lamb=0, lr=1e-3)
torch.manual_seed(0)
np.random.seed(0)
data = DataLoader(args)
args.all_ent, args.all_rel, args.eval_rel = data.all_ent, data.all_rel, data.eval_rel
KG = data.KG
split = data.triplets['test'] if dataset_dir == 'DrugBank' else data.pos_triplets['test']
pairs = torch.LongTensor(np.asarray(split)[:cli.batch, :2]).cuda()
head, tail = pairs[:, 0], pairs[:, 1]


def build(entity_feat, version='v1'):
    torch.manual_seed(0)  # same seed -> identical original modules; the gate is built after them
    args.entity_feat, args.gate_version = entity_feat, version
    return EmerGNN(data.eval_ent, data.eval_rel, args).cuda()


ok = True


def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')


base = build('none')
with torch.no_grad():
    out_a = base.enc_ht(head, tail, KG)
    out_b = base.enc_ht(head, tail, KG)
kernel_noise = (out_a - out_b).abs().max().item()
print(f'{dataset_dir} {cli.dataset}: original model run twice, max |diff| = {kernel_noise:.2e} (rspmm kernel noise)')

for version, feat in [(v, f) for v in ['v1', 'v2'] for f in ['id', 'random', 'name', 'llm']]:
    print(f'gate {version}, entity_feat={feat}')
    model = build(feat, version)
    shared = {k: v for k, v in model.state_dict().items() if not k.startswith('entity_gate.')}
    same_init = all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items())
    check('original parameters identical to the un-gated model', same_init and len(shared) == len(base.state_dict()))

    ht = torch.cat([model.Went(model.ent_kg[head]), model.Went(model.ent_kg[tail])], dim=-1)
    def all_gates():
        with torch.no_grad():
            per_layer = model.entity_gate(ht, center=model.pair_center())
        return per_layer, torch.stack([g for g in per_layer if g is not None])
    per_layer, gates = all_gates()
    check('all gates exactly 1 at init', torch.all(gates == 1).item())
    if version == 'v2':
        check('no gate on layer 0', per_layer[0] is None)

    with torch.no_grad():
        out = model.enc_ht(head, tail, KG)
    diff = (out - out_a).abs().max().item()
    check('output equals the original model', diff <= max(10 * kernel_noise, 1e-6), f'(max |diff| {diff:.2e})')

    opt = torch.optim.Adam(model.entity_gate.parameters(), lr=1e-2)
    loss = model.enc_r(model.enc_ht(head, tail, KG)).logsumexp(-1).mean()
    loss.backward()
    qgrad = sum(q.weight.grad.abs().sum().item() for q in model.entity_gate.query)
    check('query weights receive gradient', qgrad > 0, f'(sum |grad| {qgrad:.2e})')
    opt.step()
    _, gates = all_gates()
    on = model.entity_gate.mask.squeeze(1).bool()
    lo, hi = model.entity_gate.lo, model.entity_gate.hi
    check(f'gates stay within ({lo}, {hi})', bool(gates.min() > lo and gates.max() < hi))
    moved = (gates[:, on] - 1).abs().max().item()
    check('non-drug gates move after one step', moved > 0, f'(max |g-1| {moved:.2e})')
    check('drug gates stay exactly 1', torch.all(gates[:, ~on] == 1).item(), f'({(~on).sum().item()} drug entities)')

# checkpoints written before v2 existed must still load into a v1 model
old = sorted(f for f in os.listdir('saved_models') if 'gate-' in f and '-v2' not in f) if os.path.isdir('saved_models') else []
for f in old[:1]:
    feat = f.split('gate-')[1].split('_')[0]
    model = build(feat, 'v1')
    model.load_state_dict(torch.load(os.path.join('saved_models', f), map_location='cuda'))
    check(f'v1 checkpoint {f} loads into the v1 gate', True)

print('ALL PASS' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
