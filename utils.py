import torch
import numpy as np
import random

def to_gpu(var, is_gpu):
    return var.cuda() if is_gpu else var

def batchify(data, bsz, shuffle=False, is_gpu=False):

    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):

        batch = data[i*bsz:(i+1)*bsz]
        lengths = [len(x)-1 for x in batch]
        batch, lengths = length_sort(batch, lengths)

        source = [x[:-1] for x in batch]
        target = [x[1:] for x in batch]

        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        if is_gpu:
            source = source.cuda()
            target = target.cuda()

        batches.append((source, target, lengths))
    return batches

def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)

def log_line(line, log_file, is_print = True, is_line = True):
    with open(log_file, 'a') as f:
        if is_line:
            f.write(line + '\n')
        else:
            f.write(line + ' ')
    if is_print:
        if is_line:
            print(line)
        else:
            print(line, end = ' ')

def lstm_scores(ep_bleus, ep_selfbleus, ep_dists, bleus, selfbleus, dists):
    ep_bleus = ep_bleus.round(3)
    ep_selfbleus = ep_selfbleus.round(3)
    ep_dists = ep_dists.round(3)
    print("bleus : {}".format(ep_bleus))
    print("self bleus : {}".format(ep_selfbleus))
    print("dists : {}".format(ep_dists))

    ep_bleus = np.expand_dims(ep_bleus, axis=0)
    ep_selfbleus = np.expand_dims(ep_selfbleus, axis=0)
    ep_dists = np.expand_dims(ep_dists, axis=0)

    if len(bleus) == 0:
        bleus = ep_bleus
    else:
        bleus = np.append(bleus, ep_bleus, axis=0)

    if len(selfbleus) == 0:
        selfbleus = ep_selfbleus
    else:
        selfbleus = np.append(selfbleus, ep_selfbleus, axis=0)

    if len(dists) == 0:
        dists = ep_dists
    else:
        dists = np.append(dists, ep_dists, axis=0)

    return bleus, selfbleus, dists

def log_lstm_scores(bleus, selfbleus, dists, log_file):
    bleus = np.transpose(bleus)
    selfbleus = np.transpose(selfbleus)
    dists = np.transpose(dists)

    for i, bleu in enumerate(bleus):
        log_line("BLEU-{}".format(i+1), log_file)
        for ep_bleu in bleu:
            log_line(str(ep_bleu), log_file)

    for i, selfbleu in enumerate(selfbleus):
        log_line("SELF BLEU-{}".format(i+1), log_file)
        for ep_selfbleu in selfbleu:
            log_line(str(ep_selfbleu), log_file)

    for i, dist in enumerate(dists):
        log_line("DIST-{}".format(i+1), log_file)
        for ep_dist in dist:
            log_line(str(ep_dist), log_file)


