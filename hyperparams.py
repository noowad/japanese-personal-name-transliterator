import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    # train
    ## files
    parser.add_argument('--train_data_dir', default='datas/nations',
                        help="train data directory")
    parser.add_argument('--eval_data_dir', default='datas/cv/test/k0',
                        help="evaluation data directory")
    parser.add_argument('--standard_fpath', default='datas/standard_katakana',
                        help="standard katakana filepath for preprocessing")
    parser.add_argument('--non_standard_fpath', default='datas/non-standard_katakana',
                        help="non-standard katakana filepath for preprocessing")
    parser.add_argument('--oversampling_num', default=200, type=int)
    parser.add_argument('--sufficient_num', default=50, type=int)
    ## vocabulary
    parser.add_argument('--src_vocab', default='datas/eng_voca.txt',
                        help="source vocabulary file path")
    parser.add_argument('--tgt_vocab', default='datas/jap_voca.txt',
                        help="target vocabulary file path")
    parser.add_argument('--country_vocab', default='datas/nations.txt',
                        help="country vocabulary file path")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--logdir', default="log/tacotron-based", help="log directory")
    parser.add_argument('--num_epochs', default=30, type=int)

    # model
    parser.add_argument('--embed_size', default=256, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--max_len', default=25, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--encoder_banks', default=5, type=int, help="number of encoder banks")
    parser.add_argument('--dropout', default=0.3, type=float)

    # test
    parser.add_argument('--beam_width', default=10, type=int, help="beam search width")
    parser.add_argument('--ckpt', default='log/tacotron-based', help="checkpoint file path")
    parser.add_argument('--testdir', default="result/1", help="test result dir")
