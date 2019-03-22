import os
import argparse


def levenshtein_distance(sentence1, sentence2):
    '''
    levenshtein distance (edit distance) between sentence1, sentence2
    :param sentence1: sentence string
    :param sentence2: sentence string
    '''
    sentence1, sentence2 = list(sentence1), list(sentence2)

    if len(sentence1) > len(sentence2):
        sentence1, sentence2 = sentence2, sentence1
    # dynamic programming
    distances = range(len(sentence1) + 1)
    for num2, word2 in enumerate(sentence2):
        distances_ = [num2 + 1]
        for num1, word1 in enumerate(sentence1):
            if word1 == word2:
                distances_.append(distances[num1])
            else:
                distances_.append(1 + min((distances[num1], distances[num1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def calc_result(fdir):
    files = [f for f in os.listdir(fdir) if not f.startswith('.')]
    if 'UNK.txt' in files:
        files.remove('UNK.txt')
    all_correct_count = 0
    all_count = 0
    all_mrr = 0
    all_ed = 0
    all_length = 0
    nation_count = 0
    nation_over_85_count = 0
    for f in files:
        count = 0
        ed = 0
        correct_count = 0
        length = 0
        nation_count += 1
        source_names = []
        target_names = []
        candidates = []
        lines = open(fdir + '/' + f, 'r').read().splitlines()
        for line in lines:
            count += 1
            all_count += 1
            names = line.split('\t')
            source_names.append(names[0])
            target_names.append(names[1])
            candidates.append(names[2:][:5])
        rr_list = []

        for source_name, target_name, candidate in zip(source_names, target_names, candidates):
            rr = 0.
            ed += levenshtein_distance(target_name, candidate[0])
            length += len(target_name)
            for i, cand in enumerate(candidate):
                if target_name == cand:
                    correct_count += 1
                    all_correct_count += 1
                    rr = 1. / (i + 1)
                    break
            rr_list.append(rr)
        acc = float(correct_count) / count
        mrr = sum(rr_list) / len(rr_list)
        all_ed += ed
        all_mrr += mrr * len(rr_list)
        all_length += length
        cer = ed / length
        if acc >= 0.85:
            nation_over_85_count += 1
        with open(fdir + '/matome.result', 'a') as fout:
            fout.write(f[:3] + "\t" + str(correct_count) + "\t" + str(count) + "\t" + str(acc) + "\t" + str(
                mrr) + "\t" + str(cer) + "\n")
    all_acc = float(all_correct_count) / all_count
    all_mrr = str(all_mrr / all_count)
    all_cer = str(all_ed / all_length)
    with open(fdir + '/matome.result', 'a') as fout:
        fout.write(
            "ALL" + "\t" + str(all_correct_count) + "\t" + str(all_count) + "\t" + str(
                all_acc) + "\t" + all_mrr + "\t" + all_cer + "\n")
        fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))


# def data_over_50(fdir):
#     count = 0
#     correct_count = 0
#     nation_count = 0
#     nation_over_85_count = 0
#     all_mrr = 0
#     all_cer = 0
#     with open(fdir + '/over_50.result', 'w') as fout:
#         for line in open(fdir + '/matome.result', 'r').read().splitlines()[:-2]:
#             parts = line.split('\t')
#             nation = parts[0]
#             correct_num = int(parts[1])
#             data_num = int(parts[2])
#             result = float(parts[3])
#             mrr = float(parts[4])
#             cer = float(parts[5])
#             if data_num >= 50:
#                 nation_count += 1
#                 count += data_num
#                 correct_count += correct_num
#                 all_mrr += mrr * data_num
#                 all_cer += cer * data_num
#                 if result >= 0.85:
#                     nation_over_85_count += 1
#                 fout.write(line + '\n')
#     acc = str(float(correct_count) / count)
#     all_mrr = str(all_mrr / count)
#     all_cer = str(all_cer / count)
#     with open(fdir + '/over_50.result', 'a') as fout:
#         fout.write(
#             "ALL\t" + str(correct_count) + "\t" + str(count) + "\t" + acc + '\t' + all_mrr + "\t" + all_cer + "\n")
#         fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))
#
#
# def data_under_50(fdir):
#     count = 0
#     correct_count = 0
#     nation_count = 0
#     nation_over_85_count = 0
#     all_mrr = 0
#     all_cer = 0
#     with open(fdir + '/under_50.result', 'w') as fout:
#         for line in open(fdir + '/matome.result', 'r').read().splitlines()[:-2]:
#             parts = line.split('\t')
#             nation = parts[0]
#             correct_num = int(parts[1])
#             data_num = int(parts[2])
#             result = float(parts[3])
#             mrr = float(parts[4])
#             cer = float(parts[5])
#             if data_num < 50:
#                 nation_count += 1
#                 count += data_num
#                 correct_count += correct_num
#                 all_mrr += mrr * data_num
#                 all_cer += cer * data_num
#                 if result >= 0.85:
#                     nation_over_85_count += 1
#                 fout.write(line + '\n')
#     acc = str(float(correct_count) / count)
#     all_mrr = str(all_mrr / count)
#     all_cer = str(all_cer / count)
#     with open(fdir + '/under_50.result', 'a') as fout:
#         fout.write(
#             "ALL\t" + str(correct_count) + "\t" + str(count) + "\t" + acc + '\t' + all_mrr + "\t" + all_cer + "\n")
#         fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fdir', action='store', dest='fdir', type=str, default='result/1',
                        help='Enter fdir')
    par_args = parser.parse_args()
    fdir = par_args.fdir
    calc_result(fdir)
    #data_over_50(fdir)
    # data_under_50(fdir)
