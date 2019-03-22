# coding:utf-8

'''
基本的なルール
1.基本的にNon-standardにあるものはその通りに変換
2.小書きが入っているもの中でstandardに含まれていないものは小書きから大書きに変換
'''


def normalization(non_standard_list, standards, candidate):
    small_characters = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ'}
    non_standards = {}
    for non_standard in non_standard_list:
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
            if char not in standards:
                candidate = candidate.replace(pre, post)
    return candidate
