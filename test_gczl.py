# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import tqdm
from data import dataset as dset
import torchvision.models as tmodels
from models import models
import os
import itertools
import glob
# import pdb
import math
import collections

# import tensorboardX as tbx
from utils import utils
import torch.backends.cudnn as cudnn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/mit-states/', help='data root dir')
parser.add_argument('--cv_dir', default='cv/tmp/', help='dir to save checkpoints to')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')

# model parameters
parser.add_argument('--model', default='visprodNN', help='visprodNN|redwine|labelembed+|attributeop')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers for labelembed+')
parser.add_argument('--glove_init', action='store_true', default=False, help='initialize inputs with word vectors')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False, help='do not optimize input representations')

# regularizers
parser.add_argument('--lambda_aux', type=float, default=0.0)
parser.add_argument('--lambda_inv', type=float, default=0.0)
parser.add_argument('--lambda_comm', type=float, default=0.0)
parser.add_argument('--lambda_ant', type=float, default=0.0)

# optimization
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--eval_val_every', type=int, default=20)
parser.add_argument('--max_epochs', type=int, default=1000)

parser.add_argument('--test_set', default='test', help='test or val')


args = parser.parse_args()

def test(epoch):

    model.eval()

    accuracies = []
    all_attr_lab = []
    all_obj_lab = []
    all_pred = []
    pairs = testloader.dataset.pairs
    objs = testloader.dataset.objs
    attrs = testloader.dataset.attrs
    if args.test_set == 'test':
        val_pairs = testloader.dataset.test_pairs
    else:
        val_pairs = testloader.dataset.val_pairs
    train_pairs = testloader.dataset.train_pairs

    for idx, data in enumerate(testloader):
        data = [d.cuda() for d in data]
        attr_truth, obj_truth = data[1], data[2]
        _, predictions = model(data)
        print(type(predictions),predictions.keys())
        predictions, feats = predictions
        all_pred.append(predictions)
        all_attr_lab.append(attr_truth)
        all_obj_lab.append(obj_truth)

        if idx % 100 == 0:
            print('Tested {}/{}'.format(idx, len(testloader)))

    all_attr_lab = torch.cat(all_attr_lab)
    all_obj_lab = torch.cat(all_obj_lab)
    all_pair_lab = torch.LongTensor([
        pairs.index((attrs[all_attr_lab[i]], objs[all_obj_lab[i]]))
        for i in range(len(all_attr_lab))
    ])
    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))])
    all_accuracies = []

    # Calculate best unseen acc
    # put everything on cpu
    attr_truth, obj_truth = all_attr_lab.cpu(), all_obj_lab.cpu()
    pairs = list(
        zip(list(attr_truth.cpu().numpy()), list(obj_truth.cpu().numpy())))
    seen_ind = torch.LongTensor([
        i for i in range(len(attr_truth))
        if pairs[i] in evaluator_val.train_pairs
    ])
    unseen_ind = torch.LongTensor([
        i for i in range(len(attr_truth))
        if pairs[i] not in evaluator_val.train_pairs
    ])

    accuracies = []
    bias = 1e3
    args.bias = bias
    results = evaluator_val.score_model(
        all_pred_dict, all_obj_lab, bias=args.bias)
    match_stats = evaluator_val.evaluate_predictions(
        results, all_attr_lab, all_obj_lab, topk=args.topk)
    accuracies.append(match_stats)
    meanAP = 0
    _, _, _, _, _, _, open_unseen_match = match_stats
    accuracies = zip(*accuracies)
    open_unseen_match = open_unseen_match.byte()
    accuracies = list(map(torch.mean, map(torch.cat, accuracies)))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
    scores = results['scores']
    correct_scores = scores[torch.arange(scores.shape[0]), all_pair_lab][
        unseen_ind]
    max_seen_scores = results['scores'][
        unseen_ind][:, evaluator_val.seen_mask].topk(
            args.topk, dim=1)[0][:, args.topk - 1]
    unseen_score_diff = max_seen_scores - correct_scores
    correct_unseen_score_diff = unseen_score_diff[open_unseen_match] - 1e-4
    full_unseen_acc = [(
        epoch,
        attr_acc,
        obj_acc,
        closed_acc,
        open_acc,
        (open_seen_acc * open_unseen_acc)**0.5,
        0.5 * (open_seen_acc + open_unseen_acc),
        open_seen_acc,
        open_unseen_acc,
        objoracle_acc,
        meanAP,
        bias,
    )]
    print(
        '(val) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OpHM: %.4f | OpAvg: %.4f | OpSeen: %.4f | OpUnseen: %.4f  | OrO: %.4f | maP: %.4f | bias: %.3f'
        % (
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            meanAP,
            bias,
        ))

    correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
    magic_binsize = 20
    bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
    biaslist = correct_unseen_score_diff[::bias_skip]

    for bias in biaslist:
        accuracies = []
        args.bias = bias
        results = evaluator_val.score_model(
            all_pred_dict, all_obj_lab, bias=args.bias)
        match_stats = evaluator_val.evaluate_predictions(
            results, all_attr_lab, all_obj_lab, topk=args.topk)
        accuracies.append(match_stats)
        meanAP = 0

        accuracies = zip(*accuracies)
        accuracies = map(torch.mean, map(torch.cat, accuracies))
        attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
        all_accuracies.append((
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            meanAP,
            bias,
        ))

        print(
            '(val) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OpHM: %.4f | OpAvg: %.4f | OpSeen: %.4f | OpUnseen: %.4f  | OrO: %.4f | maP: %.4f | bias: %.3f'
            % (
                epoch,
                attr_acc,
                obj_acc,
                closed_acc,
                open_acc,
                (open_seen_acc * open_unseen_acc)**0.5,
                0.5 * (open_seen_acc + open_unseen_acc),
                open_seen_acc,
                open_unseen_acc,
                objoracle_acc,
                meanAP,
                bias,
            ))
    all_accuracies.extend(full_unseen_acc)
    seen_accs = np.array([a[-5].item() for a in all_accuracies])
    unseen_accs = np.array([a[-4].item() for a in all_accuracies])
    area = np.trapz(seen_accs, unseen_accs)
    print(
        '(val) E: %d | A: %.3f | O: %.3f | Cl: %.3f | AUC: %.4f | Op: %.4f | OpHM: %.4f | OpAvg: %.4f | OpSeen: %.4f | OpUnseen: %.4f  | OrO: %.4f | bias: %.3f'
        % (
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            area,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            bias,
        ))

    all_accuracies = [all_accuracies, area]
    return all_accuracies


#----------------------------------------------------------------#

#----------------------------------------------------------------#


testset = dset.CompositionDatasetActivations(root=args.data_dir, phase='test', split='compositional-split')
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.model == 'visprodNN':
    model = models.VisualProductNN(testset, args)
elif args.model == 'redwine':
    model = models.RedWine(testset, args)
elif args.model =='labelembed+':
    model = models.LabelEmbedPlus(testset, args)
elif args.model =='attributeop':
    model = models.AttributeOperator(testset, args)
model.cuda()

evaluator = models.Evaluator(testset,model)

checkpoint = torch.load(args.load)
# print(checkpoint)
model.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']
print ('loaded model from', os.path.basename(args.load))

with torch.no_grad():
    all_accuracies = test(0)
