"""DataTransformer module."""
#它用于对表格数据进行预处理，包括连续列和离散列的转换。
from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed#joblib 用于并行处理
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder#$用于数据转换。

#SpanInfo 用于存储每个转换后的列的维度和激活函数。ColumnTransformInfo 用于存储关于每列转换的详细信息，包括列名、列类型、转换器、输出信息和输出维度。
SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)


# 用于数据转换的类。
class DataTransformer(object):
    """Data Transformer.

     对连续列使用贝叶斯高斯混合模型（Bayesian GMM）进行建模，并将其归一化到 [-1, 1] 之间的标量和向量。
    对离散列使用 OneHotEncoder 进行编码。
    提供数据拟合、转换、逆转换以及将列名和值转换为 ID 的方法。
    """
    #初始化 DataTransformer 类的实例，设置最大聚类数和权重阈值。
    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                用于连续列的贝叶斯高斯混合模型（Bayesian GMM）的最大簇数，默认为 10。
            weight_threshold (float):
                高斯分布的权重阈值，用于确定是否保留某个高斯分布，默认为 0.005。
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    #为连续列训练贝叶斯高斯混合模型 _fit_continuous 和 _transform_continuous
    def _fit_continuous(self, data):#data是一列单独连续列，例如Destination Port
        """Train Bayesian GMM for continuous columns.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
           返回一个 ColumnTransformInfo 对象，包含列名、列类型、转换器、输出信息和输出维度等信息。
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        #使用gm.fit(data, column_name)对ClusterBasedNormalizer进行拟合，
        # 使其学习该连续列的数据分布特征，估计模式数量并拟合高斯混合模型。
        gm.fit(data, column_name)
        #通过sum(gm.valid_component_indicator)计算有效组件（即满足权重阈值等条件的高斯组件）的数量，赋值给num_components。
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type='continuous',
            transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components,
        )

    #离散列处理方法 _fit_discrete 和 _transform_discrete
    # 该方法用于为离散列拟合一热编码器。
    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
           返回一个 ColumnTransformInfo 对象，包含列名、列类型、转换器、输出信息和输出维度等信息。
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()#初始化编码技术
        ohe.fit(data, column_name)#使用 fit 方法对 OneHotEncoder 进行拟合。
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,#前列的名称。
            column_type='discrete',#表示离散型。
            transform=ohe,#用于转换数据的 OneHotEncoder 对象。
            output_info=[SpanInfo(num_categories, 'softmax')],#这里维度为类别数量，激活函数为 'softmax'。
            output_dimensions=num_categories,#转换后数据的维度，即类别数量。
        )

    #用于对整个数据集进行拟合，分别对连续列和离散列应用相应的转换器。
    def fit(self, raw_data, discrete_columns=()):
        #aw_data 是需要拟合的原始数据，
        #discrete_columns 是一个元组，包含所有离散列的列名，默认为空元组。

        self.output_info_list = []#用于存储每列的转换信息
        self.output_dimensions = 0#用于存储转换后数据的总维度
        self.dataframe = True#用于标记输入数据是否为 pandas.DataFrame 类型
        if not isinstance(raw_data, pd.DataFrame):#检查 raw_data 是否为 pandas.DataFrame 类型。否则则将其转化
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        #推断 raw_data 中对象类型列的数据类型，并将结果存储在 _column_raw_dtypes 属性中。
        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        #初始化一个列表，用于存储每列的转换信息。
        self._column_transform_info_list = []
        #拟合每列:
        # 循环所有列名，给每个列名区分离散还是连续，则调用 _fit_discrete 方法进行拟合；否则，调用 _fit_continuous 方法。
        for column_name in raw_data.columns:

            if column_name in discrete_columns:
                # 例如：ColumnTransformInfo(column_name='Destination Port', column_type='continuous', transform=ClusterBasedNormalizer(missing_value_generation='from_column'), output_info=[SpanInfo(dim=1, activation_fn='tanh'), SpanInfo(dim=1, activation_fn='softmax')], output_dimensions=2)
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            #将每列的转换信息添加到 output_info_list 中，并更新 output_dimensions。
            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

        #对连续列数据进行转换，将转换后的数据转换为适当的输出格式。
    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]#列名
        flattened_column = data[column_name].to_numpy().flatten()#将该列数据转换成一维数组
        data = data.assign(**{column_name: flattened_column})#并重新赋值给data中的对应列
        gm = column_transform_info.transform#将之前训练好的模型gm(高斯模型)，来对数据进行转换。
        # 经过gm后，data由[Flow Duration]列  变成了[Flow Duration.normalized,Flow Duration.component]
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))#创建一个全零的numpy数组output
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()#将转换后数据中名为{column_name}.normalized的列的值赋给output的第一列。这部分数据可能是经过归一化处理后的连续值。
        # 对于转换后数据中名为{column_name}.component的列，将其转换为整数索引index，
        # 然后在output数组中，将对应索引位置（index + 1）的值设为1.0，实现了一种类似于one - hot编码的操作。
        # 这一步是为了表示数据在不同高斯组件中的所属关系，其中每个组件对应输出中的一个维度（除了第一列）。
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

     #对离散列数据进行转换，使用拟合好的 OneHotEncoder 进行转换
    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    # 同步转换数据的方法
    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    #并行转换数据的方法,适用于较大的数据集。
    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []#初始化一个空列表 processes，用于存储将要并行执行的进程。
        #遍历包含每列转换信息的列表。
        for column_transform_info in column_transform_info_list:

            column_name = column_transform_info.column_name# 从转换信息中获取列名
            data = raw_data[[column_name]]#从 raw_data 中提取这一列的数据。
            process = None
            if column_transform_info.column_type == 'continuous':#区分列类型
                #使用 joblib 的 delayed 函数创建一个延迟执行的进程对象。
                #这个对象将在并行执行时调用相应的转换方法（_transform_continuous 或 _transform_discrete）。
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=1)(processes)

    #将原始数据转换为适合模型使用的矩阵格式
    def transform(self, raw_data):
        if not isinstance(raw_data, pd.DataFrame):#检查 raw_data 是否为 pandas.DataFrame 类型。
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)
        if raw_data.shape[0] < 500:#根据数据的大小选择同步或并行转换。如果数据行数小于500，选择同步转换。
            # 。这是因为对于较小的数据量，并行处理的开销可能会超过其带来的效率提升，而对于较大的数据量，并行转换可以显著提高处理速度。
            #调用 _synchronous_transform 方法进行转换。
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            #调用 _parallel_transform 方法进行转换。
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        return np.concatenate(column_data_list, axis=1).astype(float)

    #连续列的逆变换方法： 将转换后的连续列数据逆变换回原始格式。
    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(float)
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    # 离散列的逆变换方法：
    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    # 逆变换数据的方法：将转换后的数据逆变换回原始格式。
    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    # 将给定列名和值转换为对应的ID。
    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0#用于计数离散列的数量。
        column_id = 0#用于记录当前处理的列的索引。
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:#如果 One-Hot 编码后的数组中所有值的和为 0（即 sum(one_hot) == 0），则表示 value 在 column_name 中不存在，抛出 ValueError。
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot),
        }
