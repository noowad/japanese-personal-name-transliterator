import tensorflow as tf
from module import *
from network import *
from utils import *
from data_load import load_vocab
from params import params_to_train
from rnn_wrappers import *


class Graph():
    def __init__(self, hp, is_training=True, train_var='all'):
        self.hp = hp
        self.src_w2i, _, self.tgt_w2i, self.tgt_i2w, self.country_w2i, _ = load_vocab(hp.src_vocab,
                                                                                      hp.tgt_vocab,
                                                                                      hp.country_vocab)
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.nation = tf.placeholder(tf.int32, shape=(None, self.hp.max_len,))
            self.x = tf.placeholder(tf.int32, shape=(None, self.hp.max_len,))
            self.y = tf.placeholder(tf.int32, shape=(None, self.hp.max_len,))

            # it means sequence lengths without masking
            self.x_seq_len = tf.count_nonzero(self.x, 1, dtype=tf.int32)
            self.y_seq_len = tf.count_nonzero(self.y, 1, dtype=tf.int32)

            # Encoder
            with tf.variable_scope("nation-embed", initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
                # Embedding table
                self.nation_embeddings = embed(len(self.country_w2i), self.hp.embed_size // 2)
                self.nation_embed = tf.nn.embedding_lookup(self.nation_embeddings, self.nation)
            with tf.variable_scope("enc-embed", initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
                # Embedding table
                self.x_embeddings = embed(len(self.src_w2i), self.hp.embed_size // 2)
                self.x_embed = tf.nn.embedding_lookup(self.x_embeddings, self.x)

            self.enc_embed = tf.concat([self.x_embed, self.nation_embed], 2)
            with tf.variable_scope("encoder"):
                self.enc_outputs, self.enc_states = encode(self.enc_embed,
                                                           self.x_seq_len,
                                                           self.hp.embed_size,
                                                           self.hp.dropout,
                                                           self.hp.encoder_banks,
                                                           is_training=is_training)

            # Decoder
            with tf.variable_scope("dec-embed", initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)):
                # Embedding table
                self.dec_embeddings = embed(len(self.tgt_w2i), self.hp.embed_size)
                self.dec_embed = tf.nn.embedding_lookup(self.dec_embeddings, dec_input(self.y))

            with tf.variable_scope("decoder"):
                if is_training:
                    # Training helper
                    self.helper = tf.contrib.seq2seq.TrainingHelper(self.dec_embed,
                                                                    self.y_seq_len)
                    # Decoder for training
                    self.train_outputs, self.alignments = training_decode(self.enc_outputs,
                                                                          self.x_seq_len,
                                                                          self.helper,
                                                                          len(self.tgt_w2i),
                                                                          self.hp.embed_size,
                                                                          self.hp.dropout,
                                                                          self.hp.max_len)

                    # loss
                    # for matching length
                    self.y_ = self.y[:, :tf.reduce_max(self.y_seq_len, -1)]
                    self.istarget = tf.cast(tf.not_equal(self.y_, self.tgt_w2i["P"]), tf.float32)  # masking
                    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_,
                                                                               logits=self.train_outputs)
                    self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                    # optimizing
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(self.hp.lr)
                    self.training_variables = params_to_train(self, mode=train_var)
                    self.train_op = tf.contrib.training.create_train_op(self.mean_loss, self.optimizer,
                                                                        global_step=self.global_step,
                                                                        variables_to_train=self.training_variables)

                else:
                    # Decoder for inference (I use BeamSearchDecoder)
                    self.pred_outputs = inference_decode(self.enc_outputs,
                                                         self.x_seq_len,
                                                         self.hp.max_len,
                                                         self.dec_embeddings,
                                                         len(self.tgt_w2i),
                                                         self.hp.embed_size,
                                                         self.hp.dropout,
                                                         self.hp.beam_width,
                                                         self.tgt_w2i["S"],
                                                         self.tgt_w2i["E"])
