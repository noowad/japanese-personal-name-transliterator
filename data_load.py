import tensorflow as tf
import numpy as np
from utils import oversampling
import os
from tqdm import tqdm


# E for empty, S for end of Sentence
def load_vocab(src_fpath, tgt_fpath, country_fpath):
    def _make_dicts(fpath):
        tokens = [line for line in open(fpath, 'r').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        return token2idx, idx2token

    src_w2i, src_i2w = _make_dicts(src_fpath)
    tgt_w2i, tgt_i2w = _make_dicts(tgt_fpath)
    country_w2i, country_i2w = _make_dicts(country_fpath)
    return src_w2i, src_i2w, tgt_w2i, tgt_i2w, country_w2i, country_i2w


# text->index and padding
def encode(srcs, tgts, countries, src_dict, tgt_dict, country_dict, max_len):
    srcs_index, tgts_index, countries_index = [], [], []
    for src, tgt, country in zip(srcs, tgts, countries):
        srcs_index.append([src_dict.get(t, src_dict["U"]) for t in list(src) + ["E"]])
        tgts_index.append([tgt_dict.get(t, tgt_dict["U"]) for t in list(tgt) + ["E"]])
        countries_index.append([country_dict.get(t, country_dict["UNK"]) for t in [country] * (len(src) + 1)])

    padded_srcs_index, padded_tgts_index, padded_countries_index = [], [], []
    # zero-padding
    for src_index, tgt_index, country_index in zip(srcs_index, tgts_index, countries_index):
        padded_srcs_index.append(src_index + [src_dict["P"]] * (max_len - len(src_index)))
        padded_tgts_index.append(tgt_index + [tgt_dict["P"]] * (max_len - len(tgt_index)))
        padded_countries_index.append(country_index + [country_dict["P"]] * (max_len - len(country_index)))
    return np.array(padded_srcs_index), np.array(padded_tgts_index), np.array(padded_countries_index)


def load_data(data_dir, max_len, suffi_num, oversampling_num, src_dict, tgt_dict, country_dict, type='train'):
    srcs, tgts, input_countries, real_countries = [], [], [], []
    files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    for file in files:
        lines = open(os.path.join(data_dir, file), 'r').read().splitlines()
        if len(lines) < suffi_num:
            input_country = 'UNK'
        else:
            input_country = file[:3]
        if type == 'train':
            lines = oversampling(lines, oversampling_num)
        for line in lines:
            parts = line.split('\t')
            src = parts[0] if len(parts[0]) < max_len else parts[0][:max_len - 1]
            tgt = parts[1] if len(parts[1]) < max_len else parts[1][:max_len - 1]
            srcs.append(src)
            tgts.append(tgt)
            input_countries.append(input_country)
            real_countries.append(file[:3])
            # Encoder: token-to-idx

    srcs_index, tgts_index, countries_index = encode(srcs, tgts, input_countries, src_dict, tgt_dict, country_dict,
                                                     max_len)
    return srcs, srcs_index, tgts, tgts_index, input_countries, countries_index, real_countries
