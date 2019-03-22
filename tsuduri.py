from graph import Graph
import tensorflow as tf
from data_load import *
from hyperparams import Hparams
from utils import load_hparams
import numpy as np


class Tsuduri():
    def __init__(self):
        # hyper-params
        hparams = Hparams()
        parser = hparams.parser
        self.hp = parser.parse_args()
        load_hparams(self.hp, self.hp.ckpt)

        # token dictionary
        src_w2i, _, tgt_w2i, tgt_i2w, country_w2i, _ = load_vocab(self.hp.src_vocab,
                                                                  self.hp.tgt_vocab,
                                                                  self.hp.country_vocab)
        self.src_w2i, self.tgt_w2i, self.tgt_i2w, self.country_w2i = src_w2i, tgt_w2i, tgt_i2w, country_w2i

        # for postprocessing
        self.standards = open(self.hp.standard_fpath, 'r').read().splitlines()
        self.non_standard_list = open(self.hp.non_standard_fpath, 'r').read().splitlines()

        # model
        self.model = Graph(self.hp, is_training=False)
        self.sess = tf.InteractiveSession(graph=self.model.graph)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.hp.ckpt))

    def to_katakana(self, name, country='None', top_k=10):
        names, countries = self.preprocessing(name, country)
        names_idx, countries_idx = self.encode(names, countries)

        predicted_ids = self.sess.run(self.model.pred_outputs,
                                      {self.model.x: names_idx,
                                       self.model.nation: countries_idx}).predicted_ids[:, :, :]
        lists = []
        for ids in predicted_ids:
            candidates = []
            for pred in ids.transpose(1, 0)[:top_k]:
                candidate = "".join(self.tgt_i2w[idx] for idx in pred).split("E")[0]
                candidates.append(self.postprocessing(candidate))
            lists.append(candidates)
        return ['・'.join(i) for i in zip(*lists)]

    def encode(self, srcs, countries):
        srcs_index, couns_index = [], []
        for src, country in zip(srcs, countries):
            srcs_index.append([self.src_w2i.get(t, self.src_w2i["U"]) for t in list(src) + ["E"]])
            couns_index.append([self.country_w2i.get(t, self.country_w2i["UNK"]) for t in [country] * (len(src) + 1)])

        padded_srcs_index, padded_countries_index = [], []
        # zero-padding
        for src_index, coun_index in zip(srcs_index, couns_index):
            padded_srcs_index.append(src_index + [self.src_w2i["P"]] * (self.hp.max_len - len(src_index)))
            padded_countries_index.append(coun_index + [self.country_w2i["P"]] * (self.hp.max_len - len(coun_index)))
        return np.array(padded_srcs_index), np.array(padded_countries_index)

    def preprocessing(self, name, country):
        names = name.split(' ')
        countries = [country] * len(names)
        after_names, after_countries = [], []
        for name, country in zip(names, countries):
            name = name if len(name) < self.hp.max_len else name[:self.hp.max_len - 1]
            after_names.append(name.lower())
            after_countries.append(country.upper())
        return after_names, after_countries

    def postprocessing(self, candidate):
        small_characters = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ'}
        non_standards = {}
        for non_standard in self.non_standard_list:
            pre = non_standard.split()[0]
            post = non_standard.split()[1]
            non_standards[pre] = post
        # 1
        for pre, post in non_standards.items():
            if pre in candidate:
                candidate = candidate.replace(pre, post)

        # 2
        for pre, post in small_characters.items():
            if pre in candidate:
                char = candidate[candidate.index(pre) - 1] + candidate[candidate.index(pre)]
                if char not in self.standards:
                    candidate = candidate.replace(pre, post)
        return candidate
