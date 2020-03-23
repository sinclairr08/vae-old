import math
import numpy as np

# Zhao version
def perplexity(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        nll = np.sum([- math.log(math.pow(10.0, score)) for score, _, _ in lm.full_scores(sent, bos=True, eos=False)])
        word_count = len(words)
        total_wc += word_count
        total_nll += nll
    ppl = np.exp(total_nll / total_wc)
    return ppl

'''
# Cifka version
def get_ppl(lm, sentences):
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()

        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl
'''
