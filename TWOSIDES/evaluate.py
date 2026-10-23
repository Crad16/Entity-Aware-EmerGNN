import os
import argparse
import torch
import random
from load_data import DataLoader

from base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser for EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
parser.add_argument('--dataset', type=str, default='S1_1', help='the directory to dataset')
parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to load.')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=5, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
parser.add_argument('--seed', type=int, default=0, help='the original code always used seed 0')
parser.add_argument('--entity_feat', type=str, default='none', choices=['none', 'id', 'random', 'name', 'llm'], help='entity gate keys; none = original EmerGNN')
parser.add_argument('--gate_version', type=str, default='v1', choices=['v1', 'v2'], help='entity gate variant (see entity_gate.py)')
parser.add_argument('--gate_dim', type=int, default=32, help='entity gate query/key dimension')
parser.add_argument('--gate_lr', type=float, default=None, help='learning rate for entity gate parameters (default: same as lr)')
parser.add_argument('--emb_prefix', type=str, default='../entity_text/emb/twosides', help='prefix of the entity vector files')
parser.add_argument('--attn_mode', type=str, default='learned', choices=['learned', 'ones', 'shuffle'], help='learned: original EmerGNN; ones/shuffle: headroom checks')

class options:
    def __init__():
        pass



if __name__ == '__main__':
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
    KG = dataloader.KG
    vKG = dataloader.vKG
    tKG = dataloader.tKG
    pos_triplets, neg_triplets = dataloader.pos_triplets, dataloader.neg_triplets
    train_pos, train_neg = torch.LongTensor(pos_triplets['train']).cuda(), torch.LongTensor(neg_triplets['train']).cuda()
    valid_pos, valid_neg = torch.LongTensor(pos_triplets['valid']).cuda(), torch.LongTensor(neg_triplets['valid']).cuda()
    test_pos,  test_neg  = torch.LongTensor(pos_triplets['test']).cuda(),  torch.LongTensor(neg_triplets['test']).cuda()

    args.ent_pair = dataloader.ent_pair
    args.train_ent = list(dataloader.train_ent)

    if not os.path.exists('results'):
        os.makedirs('results')

    run_tag = args.attn_mode + ('' if args.entity_feat == 'none' else '_gate-' + args.entity_feat
                                                  + ('' if args.gate_version == 'v1' else '-' + args.gate_version))
    result_file = '%s_%s_seed%d_eval.txt' % (args.dataset, run_tag, args.seed)
    model_path = os.path.join('saved_models', '%s_%s_seed%d_%%s.pt' % (args.dataset, run_tag, args.seed))
    os.makedirs('saved_models', exist_ok=True)

    def run_model(seed):
        print('seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.dataset.startswith('S1'):
            args.lr = 0.001
            args.lamb = 0.000001
            args.n_batch = 32
            args.n_dim = 32
            args.length = 3
            args.feat = 'M'

        elif args.dataset.startswith('S2'):
            args.lr = 0.003
            args.lamb = 0.000001
            args.n_batch = 64
            args.n_dim = 64
            args.length = 3
            args.feat = 'M'

        elif args.dataset == 'S0':
            args.lr = 0.01
            args.lamb = 0.000001
            args.n_dim = 32
            args.n_batch = 32
            args.length = 3
            args.feat = 'E'

        model = BaseModel(eval_ent, eval_rel, args)
        probe_h, probe_t = test_pos[:256, 0], test_pos[:256, 1]
        def gate_log():
            return '\t' + model.model.gate_summary(probe_h, probe_t) if model.model.entity_gate is not None else ''
        if model.model.entity_gate is not None:
            init_str = 'epoch:0' + gate_log()
            print(init_str)
            with open(os.path.join('results', result_file), 'a+') as f:
                f.write(init_str + '\n')
        best_acc = -1
        for e in range(args.n_epoch):
            dataloader.shuffle_train()
            KG = dataloader.KG
            train_pos, train_neg = dataloader.train_pos, dataloader.train_neg
            model.train(train_pos, train_neg, KG)
            if (e+1) % args.epoch_per_test == 0:
                v_roc, v_pr, v_ap = model.evaluate(valid_pos, valid_neg, vKG)
                t_roc, t_pr, t_ap = model.evaluate(test_pos,  test_neg,  tKG)
                out_str = 'epoch:%d\tfeat:%s lr:%.6f lamb:%.8f n_batch:%d n_dim:%d layer:%d\t[Valid] ROC-AUC:%.4f PR-AUC:%.4f AP:%.4f\t [Test] ROC-AUC:%.4f PR-AUC:%.4f AP:%.4f' % (e+1, args.feat, args.lr, args.lamb, args.n_batch, args.n_dim, args.length, v_roc, v_pr, v_ap, t_roc, t_pr, t_ap)
                if v_pr > best_acc:
                    best_acc = v_pr
                    best_str = out_str
                    if args.save_model:
                        model.save_model(best_str, model_path % 'best')
                out_str += gate_log()
                print(out_str)
                with open(os.path.join('results', result_file), 'a+') as f:
                    f.write(out_str + '\n')
        if args.save_model:
            model.save_model('last epoch', model_path % 'last')
        print('Best results:\t' + best_str)
        with open(os.path.join('results', result_file), 'a+') as f:
            f.write('Best results:\t' + best_str + '\n\n')
        return -best_acc

    run_model(args.seed)
    

