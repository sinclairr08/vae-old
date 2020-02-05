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
