class DataIndex:
    """
    Params:

    sample_ids: list[int], 真实数据的组索引列表

    features_ids: list[int], 真实数据的特征索引列表
    """

    def __init__(self, sample_ids, features_ids):
        self.sample_ids = sample_ids
        self.features_ids = features_ids

    def __len__(self):
        return len(self.sample_ids)

    def __str__(self):
        s = "  ==== Index ====\n"
        s += f"  sample_ids: {self.sample_ids}\n"
        s += f"  features_ids: {self.features_ids}\n"
        s += "  ==== Index ===="
        return s

    def __repr__(self):
        return self.sample_ids, self.features_ids
