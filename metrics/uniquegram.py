import os
from multiprocessing import Pool

import nltk
from metrics.Metrics import Metrics
from nltk import ngrams

class UniqueGram(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'UniqueGram'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 10000
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_ng()

    def get_ng(self):
        document = self.get_reference()
        length = len(document)
        grams = list()
        grams_num=0
        for sentence in document:
            temp_grams1, temp_grams2 = self.get_gram(sentence)
            grams += temp_grams1
            grams_num += temp_grams2
        if grams_num == 0:
            return 0
        return len(set(grams))/grams_num


    def get_gram(self, tokens):
        grams = list()
        if len(tokens) < self.gram:
            return grams, len(grams)
        gram_generator = ngrams(tokens, self.gram)

        for gram in gram_generator:
            grams.append(gram)
        return grams, len(grams)

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as test_text:
                for text in test_text:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference


    def calc_ng(self, reference, hypothesis, weight):
        if len(hypothesis) < self.gram:
            return 0


    def get_ng_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_ng_parallel(reference=reference)

    def get_ng_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                result.append(pool.apply_async(self.calc_ng, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            print(type(i))
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt