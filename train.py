import tensorflow as tf
from tqdm import tqdm
from hyperparams import Hparams
from utils import save_hparams, save_variable_specs
from data_load import *
from graph import Graph
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO)

logging.info("# Hyper-params")
hparams = Hparams()
hp = hparams.parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train data")
src_w2i, _, tgt_w2i, _, country_w2i, _ = load_vocab(hp.src_vocab, hp.tgt_vocab, hp.country_vocab)
_, src_names, _, tgt_names, _, countries, _ = load_data(hp.train_data_dir,
                                                        hp.max_len,
                                                        hp.sufficient_num,
                                                        hp.oversampling_num,
                                                        src_w2i,
                                                        tgt_w2i,
                                                        country_w2i,
                                                        type='train')

logging.info("# Load model")
g = Graph(hp, is_training=True)

logging.info("# Session")
with g.graph.as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))

        data_list = list(range(len(src_names)))
        for epoch in range(1, hp.num_epochs + 1):
            logging.info("# Training epoch {}".format(epoch))
            np.random.shuffle(data_list)
            train_loss = 0
            num_batch = math.ceil(len(data_list) / hp.batch_size)
            for step in tqdm(range(int(num_batch))):
                name_ids = data_list[step * hp.batch_size:step * hp.batch_size + hp.batch_size]
                loss, gs = sess.run([g.train_op, g.global_step],
                                    {g.x: src_names[name_ids],
                                     g.y: tgt_names[name_ids],
                                     g.nation: countries[name_ids]})
                train_loss += loss
                if step % 100 == 0:
                    print('\t step:{} train_loss:{:.3f}'.format(gs, loss))
            mean_train_loss = train_loss / num_batch

            model_output = "transliterator%02dL%.3f" % (epoch, mean_train_loss)
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name)
logging.info("Done")
