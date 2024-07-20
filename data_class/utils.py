import json
import pandas as pd
import numpy as np
from typing import Optional
from data_class.RealData import RealData
from data_class.DataIndex import DataIndex


def from_csv(filename,
             label: Optional[ str ] = None,
             shuffle: bool = False,
             label_mapping: Optional[ dict ] = None
             ) -> RealData:
    data = pd.read_csv(filename)
    label = data.columns[ -1 ] if label is None else label
    y = data[ label ]
    if label_mapping is not None:
        y = y.map(label_mapping)
    y = y.to_numpy()
    data.drop(label, axis=1, inplace=True)
    continue_flags = (data.dtypes == np.float64).tolist()
    X = data.to_numpy()
    features_names = list(data.columns)
    all_feature_vals = { }
    all_label_vals = np.sort(np.unique(y))
    n_samples, n_features = X.shape
    for f_id in range(n_features):
        possible_vals = np.sort(np.unique(X[ :, f_id ]))
        all_feature_vals[ features_names[ f_id ] ] = possible_vals
    continue_flags = [ False ] * n_features if continue_flags is None else continue_flags
    data = RealData(X, y, label, all_feature_vals, all_label_vals, continue_flags)
    if shuffle:
        data.shuffle()
    return data


def from_json(filename,
              label,
              continue_flags: Optional[ list[ bool ] ] = None,
              shuffle: bool = False
              ) -> RealData:
    data = json.loads(open(filename).read())
    return from_dict(data, label, continue_flags, shuffle)


def from_list(data,
              features_names,
              label_name,
              label_idx=-1,
              continue_flags: Optional[ list[ bool ] ] = None,
              shuffle: bool = False,
              ) -> RealData:
    label_idx = len(data[ 0 ]) - 1 if label_idx == -1 else label_idx
    y = np.array([ line[ label_idx ] for line in data ])
    X = np.array([ line[ :label_idx ] + line[ label_idx + 1: ] for line in data ])
    n_samples, n_features = X.shape
    all_feature_vals = { }
    all_label_vals = np.sort(np.unique(y))
    for f_id in range(n_features):
        possible_vals = np.sort(np.unique(X[ :, f_id ]))
        all_feature_vals[ features_names[ f_id ] ] = possible_vals
    continue_flags = [ False ] * n_features if continue_flags is None else continue_flags
    data = RealData(X, y, label_name, all_feature_vals, all_label_vals, continue_flags)
    if shuffle:
        data.shuffle()
    return data


def from_numpy(data,
               features_names,
               label_name,
               label_idx=-1,
               continue_flags: Optional[ list[ bool ] ] = None,
               shuffle: bool = False,
               ) -> RealData:
    label_idx = len(data[ 0 ]) - 1 if label_idx == -1 else label_idx
    y = data[ :, label_idx ]
    X = np.concatenate([ data[ :, :label_idx ], data[ :, label_idx + 1: ] ], axis=1)
    n_samples, n_features = X.shape
    all_feature_vals = { }
    all_label_vals = np.sort(np.unique(y))
    for f_id in range(n_features):
        possible_vals = np.sort(np.unique(X[ :, f_id ]))
        all_feature_vals[ features_names[ f_id ] ] = possible_vals
    continue_flags = [ False ] * n_features if continue_flags is None else continue_flags
    data = RealData(X, y, label_name, all_feature_vals, all_label_vals, continue_flags)
    if shuffle:
        data.shuffle()
    return data


def from_dict(data,
              label,
              continue_flags: Optional[ list[ bool ] ] = None,
              shuffle: bool = False
              ) -> RealData:
    X = [ ]
    y = [ ]
    features_names = list(data.keys())
    features_names.remove(label)
    for key in data:
        if key == label:
            y = data[ key ]
        else:
            X.append(data[ key ])
    X, y = np.array(X).T, np.array(y)
    n_samples, n_features = X.shape
    all_feature_vals = { }
    all_label_vals = np.sort(np.unique(y))
    for f_id in range(n_features):
        possible_vals = np.sort(np.unique(X[ :, f_id ]))
        all_feature_vals[ features_names[ f_id ] ] = possible_vals
    continue_flags = [ False ] * n_features if continue_flags is None else continue_flags
    data = RealData(X, y, label, all_feature_vals, all_label_vals, continue_flags)
    if shuffle:
        data.shuffle()
    return data


def build_index(data: RealData) -> DataIndex:
    """
    Init full index for data
    """
    sample_ids = [ i for i in range(data.n_samples) ]
    feature_ids = [ j for j in range(data.n_features) ]
    return DataIndex(sample_ids, feature_ids)


def split_from_feature_val(
        data: RealData,
        index: DataIndex,
        feature_idx: int,
        feature_val) -> DataIndex:
    assert feature_idx in index.features_ids
    sub_feature_ids = index.features_ids
    sub_sample_ids = [ ]
    for sample_id in index.sample_ids:
        if data.X[ sample_id ][ feature_idx ] == feature_val:
            sub_sample_ids.append(sample_id)
    return DataIndex(sub_sample_ids, sub_feature_ids)


def split_feature_dict(
        data: RealData,
        index: DataIndex,
        feature_idx: int,
        continue_split_val: Optional[ float ] = None) -> dict:
    """
    :return: { feature_val: data_index }
    """
    assert feature_idx in index.features_ids
    # 离散值, 每个具体的特征值分为一类
    if not data.continue_flags[ feature_idx ]:
        feature_dict = { }
        possible_feature_vals = np.sort(np.unique(data.X[ index.sample_ids, feature_idx ]))
        for val in possible_feature_vals:
            feature_dict[ val ] = split_from_feature_val(data, index, feature_idx, val)
        return feature_dict
    # 连续值, 分为 smaller 和 bigger 两类
    feature_dict = { 'smaller': [ ], 'bigger': [ ] }
    mat = np.array(data.X[ :, feature_idx ], dtype=float)
    for sample_id in index.sample_ids:
        val = mat[ sample_id ]
        if val <= continue_split_val:
            feature_dict[ 'smaller' ].append(sample_id)
        else:
            feature_dict[ 'bigger' ].append(sample_id)
    return feature_dict


if __name__ == '__main__':
    data = [
        [ 1, 2.2, 0.6, 'y' ],
        [ 2, 1.65, 0.44, 'n' ],
        [ 1, 1.01, 0.66, 'y' ],
        [ 4, 2, 0.66, 'y' ],
    ]
    features_names = [ 'A', 'B', 'C' ]
    label_name = 'D'
    continue_flags = [ False, True, True ]
    data = from_list(data, features_names, label_name, continue_flags=continue_flags)
    index = build_index(data)
    # print(split_from_feature_val(data, index, 0, 4))
    feature_dict = split_feature_dict(data, index, 0)
    for key in feature_dict:
        print(f"feature_val: {key}, index:\n{feature_dict[ key ]}")
