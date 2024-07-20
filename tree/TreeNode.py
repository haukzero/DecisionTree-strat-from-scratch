import data_class as dc
from typing import Optional


class TreeNode:
    """
    Params:

    node_id: int, 结点编号

    index: DataIndex, 数据索引

    parent_id: int, 父结点编号

    children: list[ TreeNode ], 子结点列表

    children_ids: list[ int ], 子结点编号列表

    divided_feature_id: int, 用于划分的特征在源数据中的索引

    child_div_feature_vals: list, 划分特征后子结点代表的具体特征

    split_continue_feature_val: float, 特征为连续特征时的划分值

    final_label: 叶子结点处最后给出的标签
    """

    __global_node_id = 0

    def __init__(self):
        self.node_id = TreeNode.__global_node_id
        TreeNode.__global_node_id += 1

        self.index: Optional[ dc.DataIndex ] = None

        self.parent_id = None
        self.children = [ ]
        self.children_ids = [ ]

        self.divided_feature_id = None
        self.child_div_feature_vals = [ ]

        self.split_continue_feature_val = None

        self.final_label = None

    def info(self):
        print('====== TreeNode ======')
        print("node id:                ", self.node_id)
        print("parent node:            ", self.parent_id)
        print("children nodes:         ", self.children_ids)
        print("divided feature index:  ", self.divided_feature_id)
        print("index: ")
        print(self.index)
        print("final label:            ", self.final_label)
        print('====== TreeNode ======')

    def append(self, child_node, child_div_feature_val):
        self.children.append(child_node)
        self.children_ids.append(child_node.node_id)
        child_node.parent_id = self.node_id

        self.child_div_feature_vals.append(child_div_feature_val)

    def is_leaf(self):
        return len(self.children) == 0

    def predict_sample_label(self, sample_features: list, continue_flags: list[ bool ]):
        # 找到预测结果
        if self.is_leaf():
            return self.final_label
        # 测试样本在当前结点代表的特征上的取值
        sample_val = sample_features[ self.divided_feature_id ]
        # 离散值
        if not continue_flags[ self.divided_feature_id ]:
            child_idx = self.child_div_feature_vals.index(sample_val)
            next_node = self.children[ child_idx ]
            return next_node.predict_sample_label(sample_features, continue_flags)
        # 连续值
        if sample_val <= self.split_continue_feature_val:
            return self.children[ 0 ].predict_sample_label(sample_features, continue_flags)
        return self.children[ 1 ].predict_sample_label(sample_features, continue_flags)

    def dfs(self):
        self.info()
        for child in self.children:
            child.dfs()

    def bfs(self):
        que = [ self ]
        i = 0
        while i < len(que):
            que[ i ].info()
            que.extend(que[ i ].children)
            i += 1

    def clear_id(self):
        TreeNode.__global_node_id = 0


if __name__ == '__main__':
    root = TreeNode()
    root.append(TreeNode(), 'child_1')
    root.append(TreeNode(), 'child_2')
    root.dfs()
