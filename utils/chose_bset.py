import numpy as np
import data_class as dc
from typing import Tuple
from utils import evaluate


def _entropy_method(data: dc.RealData, index: dc.DataIndex) -> Tuple[int, float]:
    max_info_gain = -1
    best_a = 0
    best_t = None
    for a in index.features_ids:
        if not data.continue_flags[a]:
            gain, _ = evaluate.infoGain_dis(data, index, a)
            if gain > max_info_gain:
                max_info_gain = gain
                best_a = a
        else:
            gain, t = evaluate.infoGain_con(data, index, a)
            if gain > max_info_gain:
                max_info_gain = gain
                best_a = a
                best_t = t
    return best_a, best_t


def _ratio_method(data: dc.RealData, index: dc.DataIndex) -> Tuple[int, None]:
    max_gain_ratio = -1
    best_a = 0
    for a in index.features_ids:
        ratio, _ = evaluate.Gain_ratio(data, index, a)
        if ratio > max_gain_ratio:
            max_gain_ratio = ratio
            best_a = a
    return best_a, None


def _gini_method(data: dc.RealData, index: dc.DataIndex) -> Tuple[int, None]:
    min_gini_index = 0x7fff
    best_a = 0
    for a in index.features_ids:
        gini, _ = evaluate.Gini_index(data, index, a)
        if gini < min_gini_index:
            min_gini_index = gini
            best_a = a
    return best_a, None


def divide_feature_method(method: str):
    """
    :param method: ['entropy', 'ratio', 'gini']
    :return: func
    """
    assert method in [ 'entropy', 'ratio', 'gini' ]
    if method == 'entropy':
        return _entropy_method
    elif method == 'gain_ratio':
        return _ratio_method
    return _gini_method


def max_num_label(data: dc.RealData, index: dc.DataIndex):
    uni_label, counts = np.unique(data.y[index.sample_ids], return_counts=True)
    max_cnt = np.argmax(counts)
    return uni_label[max_cnt]


if __name__ == '__main__':
    data = [
        [ 1, 1, 'y' ],
        [ 1, 1, 'y' ],
        [ 1, 0, 'n' ],
        [ 1, 0, 'n' ],
        [ 0, 0, 'n' ]
    ]
    features_names = ['A', 'B']
    label_name = 'C'
    data = dc.from_list(data, features_names, label_name)
    index = dc.build_index(data)
    method = divide_feature_method('gini')
    print(method(data, index))
