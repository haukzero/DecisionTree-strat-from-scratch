import numpy as np
from typing import Optional
from data_class.DataIndex import DataIndex


class RealData:
    """
    Params:

    n_sample: int, 数据组数

    n_features: int, 特征种类数

    X: np.ndarray, 特征

    y: np.ndarray, 标签

    label: 标签名

    all_feature_vals: dict[str, list], 以字典形式存储每种特征在给定数据集内的所有可能取值

    all_label_vals: list, 以列表形式存储标签在给定数据集内的所有可能取值

    continue_flags: list[ bool ], 记录特征是否连续的标记列表
    """

    def __init__(self, X, y, label, all_feature_vals, all_label_vals, continue_flags=None):
        self.n_samples = X.shape[ 0 ]
        self.n_features = X.shape[ 1 ]
        self.X = X
        self.y = y
        self.label = label
        self.all_feature_vals = all_feature_vals
        self.all_label_vals = all_label_vals
        self.continue_flags = [ False ] * self.n_features if continue_flags is None else continue_flags

    def __len__(self):
        return self.n_samples

    def __str__(self):
        s = " ==== Data ====\n"
        s += f" n_samples: {self.n_samples}\n"
        s += f" n_features: {self.n_features}\n"
        s += f" X: {self.X}\n"
        s += f" y: {self.y}\n"
        s += f" label: {self.label}\n"
        s += f" all_feature_vals:\n"
        for key in self.all_feature_vals:
            s += f"  {key}: {self.all_feature_vals[ key ]}\n"
        s += f" all_label_vals: {self.all_label_vals}\n"
        s += f" continue_flags: {self.continue_flags}\n"
        s += " ==== Data ====\n"
        return s

    def __repr__(self):
        return self.__str__()

    def __get_train_data(self, data, features_names, label_name, continue_flags):
        y = np.array([ line[ -1 ] for line in data ])
        X = np.array([ line[ :-1 ] for line in data ])
        n_samples, n_features = X.shape
        all_feature_vals = { }
        all_label_vals = np.sort(np.unique(y))
        for f_id in range(n_features):
            possible_vals = np.sort(np.unique(X[ :, f_id ]))
            all_feature_vals[ features_names[ f_id ] ] = possible_vals
        return RealData(X, y, label_name, all_feature_vals, all_label_vals, continue_flags)

    def same_in_features(self, index: DataIndex) -> bool:
        """
        判断数据在给定索引内是否具有相同的特征值
        """
        if len(index) <= 1:
            return True
        # 一列一列下来判断特征值
        for j in index.features_ids:
            base = self.X[ index.sample_ids[ 0 ], j ]
            for i in range(1, len(index)):
                tmp_feature = self.X[ index.sample_ids[ i ], j ]
                if tmp_feature != base:
                    return False
        return True

    def same_in_label(self, index: DataIndex):
        """
        判断数据在给定索引内是否具有相同的标签值, 只需判断里面的标签值是否只有一种
        """
        if len(index) <= 1:
            return True
        label_list = [ ]
        for i in index.sample_ids:
            if self.y[ i ] not in label_list:
                label_list.append(self.y[ i ])
        return len(label_list) == 1

    def shuffle(self, seed: Optional[ int ] = None):
        np.random.seed(seed)
        shuffle_idx = np.random.permutation(self.n_samples)
        self.X = self.X[ shuffle_idx ]
        self.y = self.y[ shuffle_idx ]

    def split_train_test(self,
                         train_ratio: float = 0.8,
                         shuffle: bool = True,
                         seed: Optional[ int ] = None
                         ):
        """
        :return: (train_data: RealData, test_X: list[ list ], test_y: list)
        """
        if shuffle:
            self.shuffle(seed)

        n_train = int(self.n_samples * train_ratio)
        test_X, test_y = self.X[ n_train: ], self.y[ n_train: ]
        train_X, train_y = self.X[ : n_train ], self.y[ : n_train ]
        train_X, train_y = train_X.tolist(), train_y.tolist()
        train_data = [ train_X[ i ] + [ train_y[ i ] ] for i in range(n_train) ]
        train_data = self.__get_train_data(train_data,
                                           list(self.all_feature_vals.keys()),
                                           self.label,
                                           continue_flags=self.continue_flags)
        return train_data, test_X, test_y
