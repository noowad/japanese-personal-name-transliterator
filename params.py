# coding:utf-8
import tensorflow as tf
import os


# 学習したいparameterリストを出力
def params_to_train(g=None, mode='all'):
    params = []
    with g.graph.as_default():
        if mode == 'all':
            for param in tf.trainable_variables():
                params.append(param)
        if mode == 'nation':
            for param in tf.trainable_variables():
                if 'nation-embed' in str(param):
                    params.append(param)
        if mode == 'character_embedding':
            for param in tf.trainable_variables():
                if 'enc-embed' in str(param):
                    params.append(param)
        if mode == 'encoder':
            for param in tf.trainable_variables():
                if ('enc-embed' in str(param)) or ('encoder' in str(param)):
                    params.append(param)
        # encoderのembeddingからprenetまで
        if mode == 'prenet':
            for param in tf.trainable_variables():
                if ('enc-embed' in str(param)) or ('encoder/prenet' in str(param)):
                    params.append(param)
        # encoderのembeddingからcbhgモジュールまで
        if mode == 'cbhg':
            for param in tf.trainable_variables():
                if ('enc-embed' in str(param)) \
                        or ('encoder/prenet' in str(param)) \
                        or ('encoder/conv1d_banks' in str(param)) \
                        or ('encoder/conv1d_1' in str(params)) \
                        or ('encoder/conv1d_2' in str(params)) \
                        or ('encoder/norm1' in str(params)) \
                        or ('encoder/norm2' in str(params)):
                    params.append(param)
        if mode == 'decoder':
            for param in tf.trainable_variables():
                params.append(param)
            for param in tf.trainable_variables():
                if ('enc-embed' in str(param)) or ('encoder' in str(params)):
                    params.remove(param)
    return params
