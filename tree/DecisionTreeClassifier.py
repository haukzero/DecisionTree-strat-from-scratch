import numpy as np
import data_class as dc
import utils
from tree.TreeNode import TreeNode
from typing import Union


class DecisionTreeClassifier:
    """
    Params:

    root: TreeNode, 决策树的根节点

    features_names: list[ str ], 特征名称列表, 方便后面画图

    continue_flags: list[ bool ], 记录特征是否连续的标记列表, 方便后面生成决策树与画图
    """

    def __init__(self):
        self.root = TreeNode()
        self.features_names = None
        self.continue_flags = None

    # 最核心的生成部分
    def _generate(self,
                  data: dc.RealData,
                  cur_node: TreeNode,
                  divide_feature_method):
        # 数据第一次传入, 初始化
        if self.features_names is None:
            self.features_names = list(data.all_feature_vals.keys())
            self.continue_flags = data.continue_flags

        # 数据标签为同一类别, 说明分好类了
        if data.same_in_label(cur_node.index):
            cur_node.final_label = data.y[ cur_node.index.sample_ids[ 0 ] ]
            return

        most_possible_label = utils.max_num_label(data, cur_node.index)

        # 特征集为空 或 数据在给定特征集上取值相同, 直接把最可能取到的标签值设为最终结果
        if (not len(cur_node.index)) or \
                data.same_in_features(cur_node.index):
            cur_node.final_label = most_possible_label
            return

        # 获取 最佳拆分特征 及 对应取值(连续值特有)
        best_div_feature_idx, best_con_div_val = divide_feature_method(data, cur_node.index)
        cur_node.divided_feature_id = best_div_feature_idx
        best_div_feature_name = list(data.all_feature_vals.keys())[ best_div_feature_idx ]

        # 如果当前选取的特征值为连续值, 则可以重复利用, 否则要丢弃不用
        sub_available_feature_ids = cur_node.index.features_ids.copy()
        if not data.continue_flags[ best_div_feature_idx ]:
            sub_available_feature_ids.remove(best_div_feature_idx)

        feature_dict = dc.split_feature_dict(data,
                                             cur_node.index,
                                             best_div_feature_idx,
                                             best_con_div_val)

        # 连续特征, 只有 smaller / bigger 两结点
        if data.continue_flags[ best_div_feature_idx ]:
            cur_node.split_continue_feature_val = best_con_div_val
            # smaller
            smaller_feature_ids = sub_available_feature_ids.copy()
            smaller_sample_ids = feature_dict[ 'smaller' ]
            smaller_index = dc.DataIndex(smaller_sample_ids, smaller_feature_ids)
            smaller_node = TreeNode()
            smaller_node.index = smaller_index
            cur_node.append(smaller_node, 'smaller')
            self._generate(data, smaller_node, divide_feature_method)
            # bigger
            bigger_feature_ids = sub_available_feature_ids.copy()
            bigger_sample_ids = feature_dict[ 'bigger' ]
            bigger_index = dc.DataIndex(bigger_sample_ids, bigger_feature_ids)
            bigger_node = TreeNode()
            bigger_node.index = bigger_index
            cur_node.append(bigger_node, 'bigger')
            self._generate(data, bigger_node, divide_feature_method)

        # 离散特征, 每个具体特征都有对应子结点
        else:
            for feature_val in data.all_feature_vals[ best_div_feature_name ]:
                sub_node = TreeNode()
                sub_feature_ids = sub_available_feature_ids.copy()
                sub_sample_ids = [ ]
                sub_index = dc.DataIndex(sub_sample_ids, sub_feature_ids)
                sub_node.index = sub_index
                cur_node.append(sub_node, feature_val)
                # 若拆分出的字典键值中有 feature_val
                if feature_val in feature_dict.keys():
                    sub_node.index.sample_ids = feature_dict[ feature_val ].sample_ids
                    self._generate(data, sub_node, divide_feature_method)
                # 若拆分出的字典键值中没出现 feature_val, 即当前分出的数据中没有这种情况
                # 将其设为叶子节点, final_label 设为最可能的值
                else:
                    sub_node.final_label = most_possible_label

    def fit(self, data: dc.RealData, divide_feature_method):
        self.root.index = dc.build_index(data)
        self._generate(data, self.root, divide_feature_method)

    def predict(self, X: Union[ list, np.ndarray ]) -> list:
        labels = [ ]
        for sample in X:
            labels.append(self.root.predict_sample_label(sample, self.continue_flags))
        return labels

    def score(self, X, y) -> float:
        pred = self.predict(X)
        n = len(y)
        cnt = 0
        for i in range(0, n):
            if pred[ i ] == y[ i ]:
                cnt += 1
        return cnt / n

    def dfs_show(self):
        self.root.dfs()

    def bfs_show(self):
        self.root.bfs()

    def clear(self):
        self.root = TreeNode()
        self.root.clear_id()
        self.root.node_id = 0
        self.features_names = None
        self.continue_flags = None


if __name__ == '__main__':
    data = [ [ 1, 1, "yes" ],
             [ 1, 1, "yes" ],
             [ 1, 0, "no" ],
             [ 0, 1, "no" ],
             [ 0, 1, "no" ],
             [ 0, 0, 'no' ] ]
    features_names = [ 'A', 'B' ]
    label = 'C'
    data = dc.from_list(data, features_names, label)
    index = dc.build_index(data)
    method = utils.divide_feature_method('entropy')
    dt = DecisionTreeClassifier()
    dt.fit(data, method)
    dt.bfs_show()

    # test_x = [
    #     [ 1, 1 ],
    #     [ 1, 0 ],
    #     [ 0, 1 ],
    #     [ 0, 0 ]
    # ]
    # test_y = [ 'yes', 'no', 'no', 'no' ]
    # print(dt.score(test_x, test_y))
