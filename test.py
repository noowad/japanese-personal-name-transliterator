import os

from graph import Graph
import tensorflow as tf
from data_load import *
from hyperparams import Hparams
from utils import load_hparams
import logging
import postprocessing

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test data")
src_w2i, _, tgt_w2i, tgt_i2w, country_w2i, _ = load_vocab(hp.src_vocab, hp.tgt_vocab, hp.country_vocab)
src_names, idx_src_names, tgt_names, _, countries, idx_countries, real_countries = load_data(hp.eval_data_dir,
                                                                                             hp.max_len,
                                                                                             hp.sufficient_num,
                                                                                             hp.oversampling_num,
                                                                                             src_w2i,
                                                                                             tgt_w2i,
                                                                                             country_w2i,
                                                                                             type='test')
# for postprocessing
standards = open(hp.standard_fpath, 'r').read().splitlines()
non_standard_list = open(hp.non_standard_fpath, 'r').read().splitlines()

logging.info("# Load model")
g = Graph(hp, is_training=False)

if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
logging.info("# Session")

with g.graph.as_default(), tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_  # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)
    count = 0
    for i in range(0, len(src_names), hp.batch_size):
        batch_src_names = src_names[i:i + hp.batch_size]
        batch_idx_src_names = idx_src_names[i:i + hp.batch_size]
        batch_tgt_names = tgt_names[i:i + hp.batch_size]
        batch_countries = real_countries[i:i + hp.batch_size]
        batch_idx_countries = idx_countries[i:i + hp.batch_size]
        batch_predicted_ids = sess.run(g.pred_outputs,
                                       {g.x: batch_idx_src_names,
                                        g.nation: batch_idx_countries}).predicted_ids[:, :, :]
        for src, tgt, ids, country in zip(batch_src_names, batch_tgt_names, batch_predicted_ids, batch_countries):
            count += 1
            predicted_ids = ids.transpose(1, 0)
            print(src, tgt)
            with open(os.path.join(hp.testdir, country + '.txt'), 'a') as fout:
                fout.write(src + '\t' + tgt)
                for pred in predicted_ids:
                    candidate = "".join(tgt_i2w[idx] for idx in pred).split("E")[0]
                    candidate = postprocessing.normalization(non_standard_list,standards,candidate)
                    fout.write('\t' + candidate)
                fout.write('\n')
