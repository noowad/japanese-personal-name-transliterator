import tensorflow as tf
import os
import random
import json
import logging


def dec_input(labels):
    x = tf.cast(tf.fill([tf.shape(labels)[0], 1], 2), dtype=tf.int32)
    return tf.concat([x, labels[:, :-1]], 1)





def oversampling(data, oversampling_num):
    random.seed(1)
    if len(data) >= oversampling_num:
        return data
    else:
        return data + random.choices(data, k=oversampling_num - len(data))


def preprocess(katakana, standards, non_standards):
    '''Katakana standardization.
    You can also use this standardization in postprocessing
    '''
    small_characters = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ'}
    # 1
    for pre, post in non_standards.items():
        if pre in katakana:
            katakana = katakana.replace(pre, post)
    # 2
    for pre, post in small_characters.items():
        if pre in katakana:
            char = katakana[katakana.index(pre) - 1] + katakana[katakana.index(pre)]
            if char not in standards:
                katakana = katakana.replace(pre, post)
    return katakana


def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)


def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v


def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''

    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *= shp[d]
        return size

    params, num_params = [], 0
    for v in tf.trainable_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")
