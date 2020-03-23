import os

from metrics.perplexity import perplexity
from utils import log_line

def train_lm(data_path,save_path, maxlen, sample_method, dict, N, epoch, model, log_file):

    max_indices = model.sample(epoch, sample_num=100, maxlen=maxlen, idx2word=dict.idx2word, log_file=None, save_path=save_path,
                               sample_method = sample_method, return_index=True)

    lm_file = os.path.join(save_path, 'epoch_' + str(epoch) + "_lm.txt")
    lm_arpa = os.path.join(save_path, 'epoch_' + str(epoch) + "_lm.arpa")
    with open(lm_file, "w") as f:

        for word in dict.word2idx.keys():
            f.write(word + '\n')

        for idx in max_indices:
            words = [dict.idx2word[x] for x in idx]

            truncated_sent = []

            for w in words:
                if w!= '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+'\n')

        try:
            rev_lm = train_ngram_lm(kenlm_path='./kenlm',
                                    data_path=lm_file,
                                    output_path=lm_arpa,
                                    N = N)
            with open(os.path.join(data_path, 'test.txt'), 'r') as f:
                lines = f.readlines()
                lines = list(map(lambda x:x.lower(), lines))
            sentences = [l.replace('\n', '') for l in lines]
            rev_ppl = perplexity(rev_lm, sentences)

        except:
            print("reverse ppl error: it maybe the generated files aren't valid to obtain an LM")
            rev_ppl = 1e15

        for_lm = train_ngram_lm(kenlm_path='./kenlm',
                                    data_path=os.path.join(data_path, 'train.txt'),
                                    output_path=lm_arpa,
                                    N = N

        )
        with open(lm_file, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.lower(), lines))
        sentences = [l.replace('\n', '') for l in lines]
        for_ppl = perplexity(for_lm, sentences)

        log_line("Epoch {} REV PPL : {:.4f} FOR PPL : {:.4f}".format(
            epoch, rev_ppl, for_ppl), log_file, is_print=True)

        return rev_ppl, for_ppl

def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)

    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    import kenlm
    # create language model
    model = kenlm.LanguageModel(output_path)

    return model
