import math
import numpy as np
import data_class as dc
from typing import Tuple


def _np_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    P = counts / labels.shape[ 0 ]
    return -P @ np.log2(P).T


def Entropy(data: dc.RealData, index: dc.DataIndex) -> float:
    r"""
    $Ent(D) = - \sum p \log_2 p$
    """
    label_dict = { }
    for label in data.y[ index.sample_ids ]:
        if label not in label_dict:
            label_dict[ label ] = 0
        label_dict[ label ] += 1
    entropy = 0
    for label in label_dict:
        p = label_dict[ label ] / len(index)
        entropy -= p * math.log2(p)
    return entropy


def infoGain_dis(data: dc.RealData, index: dc.DataIndex, a: int) -> Tuple[ float, None ]:
    r"""
    $Gain(D, a) = Ent(D) - \sum \frac{|D^v|}{|D|} Ent(D^v)$
    """
    assert a in index.features_ids and not data.continue_flags[ a ]
    base_entropy = Entropy(data, index)
    feature_dict = dc.split_feature_dict(data, index, a)
    for val in feature_dict:
        base_entropy -= len(feature_dict[ val ]) / len(index) * Entropy(data, feature_dict[ val ])
    return base_entropy, None


def _infoGain_con_one(data: dc.RealData, index: dc.DataIndex, a: int, t: float) -> float:
    assert a in index.features_ids and data.continue_flags[ a ]
    base_entropy = Entropy(data, index)
    mask = np.array(data.X[ index.sample_ids, a ], dtype=float) <= t
    smaller = data.y[ index.sample_ids ][ mask ]
    bigger = data.y[ index.sample_ids ][ ~mask ]
    s_ent = _np_entropy(smaller)
    b_ent = _np_entropy(bigger)
    gain = base_entropy - (smaller.shape[ 0 ] * s_ent + bigger.shape[ 0 ] * b_ent) / len(index)
    return gain


def infoGain_con(data: dc.RealData, index: dc.DataIndex, a: int) -> Tuple[ float, float ]:
    r"""
    $Gain(D, a) = Ent(D) - \sum \frac{|D^v|}{|D|} Ent(D^v)$
    """
    assert a in index.features_ids and data.continue_flags[ a ]
    if len(index) == 1:
        return (_infoGain_con_one(data, index, a, data.X[ index.sample_ids, a ]),
                data.X[ index.sample_ids, a ])
    feature_vals = np.sort(np.array(data.X[ index.sample_ids, a ], dtype=float))
    info_gain = -1
    best_feature_val = (feature_vals[ 0 ] + feature_vals[ 1 ]) / 2
    for i in range(0, len(feature_vals) - 1):
        t = float((feature_vals[ i ] + feature_vals[ i + 1 ]) / 2)
        gain = _infoGain_con_one(data, index, a, t)
        if gain > info_gain:
            info_gain = gain
            best_feature_val = t
    return info_gain, best_feature_val


def _IV(data: dc.RealData, index: dc.DataIndex, a: int) -> float:
    r"""
    $IV(D, a) = - \sum \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}$
    """
    assert a in index.features_ids
    iv = 0
    feature_dict = dc.split_feature_dict(data, index, a)
    for val in feature_dict:
        p = len(feature_dict[ val ]) / len(index)
        iv -= p * math.log2(p)
    return iv


def Gain_ratio(data: dc.RealData, index: dc.DataIndex, a: int) -> Tuple[ float, None ]:
    r"""
    $Gain_ratio(D, a) = \frac{Gain(D, a)}{IV(D, a)}$
    """
    assert a in index.features_ids and not data.continue_flags[ a ]
    gain, _ = infoGain_dis(data, index, a)
    iv = _IV(data, index, a)
    return gain / iv, None


def _Gini(data: dc.RealData, index: dc.DataIndex) -> float:
    r"""
    $Gini(D) = 1 - \sum p^2$
    """
    label_dict = { }
    for label in data.y[ index.sample_ids ]:
        if label not in label_dict:
            label_dict[ label ] = 0
        label_dict[ label ] += 1
    gini = 1
    for label in label_dict:
        p = label_dict[ label ] / len(index)
        gini -= p * p
    return gini


def Gini_index(data: dc.RealData, index: dc.DataIndex, a: int) -> Tuple[ float, None ]:
    r"""
    Gini_index(D, a) = \sum \frac{|D^v|}{|D|} Gini(D^v)$
    """
    assert a in index.features_ids and not data.continue_flags[ a ]
    gini_index = 0
    feature_dict = dc.split_feature_dict(data, index, a)
    for val in feature_dict:
        Dv_gini = _Gini(data, feature_dict[ val ])
        gini_index += len(feature_dict[ val ]) * Dv_gini / len(index)
    return gini_index, None


if __name__ == '__main__':
    datasets = [
        [ 1, 1, 'y' ],
        [ 1, 0, 'n' ],
        [ 0, 1, 'n' ],
        [ 0, 0, 'n' ]
    ]
    features_names = [ 'A', 'B' ]
    label = 'C'
    data = dc.from_list(datasets, features_names, label)
    index = dc.build_index(data)
    print(Gini_index(data, index, a=0))
