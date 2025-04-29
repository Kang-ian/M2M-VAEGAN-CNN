"""DataSampler module."""

import numpy as np

#采样条件向量和相应的训练数据
class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""
   #接收原始数据 data、输出信息 output_info（描述数据列的特性）和 log_frequency（一个布尔值，指示是否使用对数频率来计算类别概率）。
    def __init__(self, data, output_info, log_frequency):
        self._data_length = len(data)
        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'
        #统计离散列的数量
        n_discrete_columns = sum([
            1 for column_info in output_info if is_discrete_column(column_info)
        ])
        #用于存储每个离散列的起始索引
        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')

        #用于存储每个类别在每个离散列中的行索引
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols 计算每个离散列中每个类别的频率，并根据 log_frequency 决定是否使用对数频率
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim for column_info in output_info if is_discrete_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    #根据给定的离散列 ID，从该列的类别概率分布中随机选择一个索引。
    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    #用于生成训练过程中使用的条件向量  该类负责从训练数据中采样，并生成用于条件生成的数据。
    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
           cond：条件向量，形状为(batch, #categories)，表示每个样本在不同类别中的状态。
            mask：掩码向量，形状为(batch, #discrete columns)，是一个一热编码向量，指示选定的离散列。
            discrete_column_id：整数表示的掩码，形状为(batch,)，表示每个样本选择的离散列的索引。
            category_id_in_col：在选定的离散列中选择的类别，形状为(batch,)。
        """
        #检查离散列的数量  为了规律但不均匀的找到值
        if self._n_discrete_columns == 0:
            return None
        #随机选择离散列：
        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)
        #初始化条件向量和掩码：
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        #设置掩码：
        mask[np.arange(batch), discrete_column_id] = 1
        #随机选择类别：
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        #计算类别ID：
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        #在cond中，将每个样本对应的类别ID位置设置为1。
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    #这个方法生成用于生成的条件向量，它使用原始频率而不是对数频率。
    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        cond[np.arange(batch), col_idxs] = 1

        return cond

    #根据采样的条件向量从原始训练数据中采样数据。
    def sample_data(self, data, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Args:
            data:
                The training data.

        Returns:
            n:
                n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return data[idx]

    #返回条件向量的维度
    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    #根据给定的条件列信息生成条件向量。
    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
