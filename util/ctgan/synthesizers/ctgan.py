"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
from util.ctgan.errors import InvalidDataError
from util.ctgan.data_sampler import DataSampler
from util.ctgan.data_transformer import DataTransformer
from util.ctgan.synthesizers.base import BaseSynthesizer, random_state


class Discriminator(Module):
    """Discriminator for the CTGAN."""
    """pac
    增强判别器的稳定性：通过同时考虑多个样本，判别器可以更好地学习数据的分布，从而提高其稳定性。
提高判别器的判别能力：PAC允许判别器在更大的上下文中评估单个样本，这有助于提高其区分真实和假数据的能力。
减少模式崩溃：通过考虑样本之间的关系，PAC有助于生成器产生更多样化的数据，减少模式崩溃的风险。"""
    def __init__(self, input_dim, discriminator_dim, pac=10):
        """input_dim（输入数据的维度）
        discriminator_dim（判别器层的维度列表）
        pac（Pairwise Adversarial Conditioning，一对对抗性条件，用于将多个样本组合在一起进行判别）。"""
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    #梯度惩罚计算  WGAN中
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """self不同 alpha有pac gradient_view有点区别"""
        #alpha是一个随机张量，用于在真实数据和生成数据之间插值。
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)#pac 确保 alpha 的第一个维度是批次大小除以 pac，这意味着每个 alpha 将用于 pac 个样本
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        #interpolates是插值后的数据。这个插值过程可以看作是在真实数据和生成数据之间的直线上随机采样点
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        #disc_interpolates是判别器对插值数据的输出。 用于计算梯度
        disc_interpolates = self(interpolates)
        #gradients是插值数据相对于判别器输出的梯度。
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        #gradients_view是梯度的范数减1。
        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        #gradient_penalty是梯度惩罚项，用于正则化判别器。
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))

#残差层
#残差层的设计目的是帮助解决深度神经网络中的梯度消失问题，通过添加输入和输出来允许网络学习输入和输出之间的残差映射。
class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        #将ReLU激活函数的输出与原始输入input_在维度1（特征维度）上进行拼接（torch.cat）。
        return torch.cat([out, input_], dim=1)

""" self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)"""
class Generator(Module):
    """Generator for the CTGAN."""
    """embedding_dim：输入向量的维度，通常是随机噪声和条件向量的组合。
    generator_dim：生成器内部层的维度列表，这些维度定义了残差层的大小。
    data_dim：输出数据的维度，即生成数据的大小。"""
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,#嵌入维度
        generator_dim=(256, 256),#生成器和判别器的层大小
        discriminator_dim=(256, 256),
        generator_lr=2e-4,#学习率
        generator_decay=1e-6,#权重衰减
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,#批次
        discriminator_steps=5,#判别器更新步数
        log_frequency=True,#是否使用类别的对数频率进行条件采样
        verbose=False,#是否打印进度
        epochs=300,#训练周期数
        pac=10,#PAC
        cuda=True,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    ##实现了Gumbel-Softmax技巧，用于处理早期PyTorch版本中的不稳定性。
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    #用于对生成器输出应用适当激活函数的关键部分
    def _apply_activate(self, data):
        #对生成器输出应用适当的激活函数。
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                #如果激活函数是 tanh，则使用 torch.tanh 对 data 中相应的列应用 tanh 激活函数，并将结果添加到 data_t。
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                #如果激活函数是 softmax，则调用 self._gumbel_softmax 方法（如之前分析的）对 data 中相应的列应用 Gumbel-Softmax 激活函数，并将结果添加到 data_t。
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    #如果激活函数既不是 tanh 也不是 softmax，则抛出 ValueError 异常，提示遇到了意外的激活函数。
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
# 将 data_t 列表中的所有处理过的数据列在维度1（特征维度）上进行拼接，返回拼接后的数据张量。
        return torch.cat(data_t, dim=1)


        #计算固定离散列的交叉熵损失。

    #用于计算固定离散列的交叉熵损失。
    def _cond_loss(self, data, c, m):
        """data：生成器产生的数据，形状为 [batch_size, data_dim]，其中 data_dim 是数据的维度。
        c：真实的条件数据，通常包含离散特征的真实值，形状为 [batch_size, data_dim]。
        m：掩码或权重，用于对不同样本的损失进行加权，形状为 [batch_size, 1] 或 [batch_size]。"""
        loss = []
        st = 0#st 和 st_c，分别用于跟踪 data 和 c 中的当前位置。
        st_c = 0
        #这是一个包含数据列信息的列表。每个元素 column_info 包含一列的信息，span_info 包含列中每个跨度的信息。
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    #检查是否是离散列 且激活函数为softmax
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    #计算这部分数据的交叉熵损失
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013 使用 torch.stack 将 loss 列表中的所有损失值堆叠成一个张量

        return (loss * m).sum() / data.size()[0]#平均交叉熵损失

        #验证discrete_columns是否存在于train_data中。
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)#检查是否指定的里散列在train中
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )
        #数据转换
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        """不同于TVAE的地方"""
        #数据采样     用于采样条件向量和相应的训练数据。
        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        #ata_dim 被设置为转换器输出的维度，这代表了转换后数据的特征数量。
        data_dim = self._transformer.output_dimensions

        #初始化生成器和判别器
        #self._embedding_dim + self._data_sampler.dim_cond_vec()输入向量的维度，这个向量通常是随机噪声和条件向量的组合
        #generator_dim：生成器内部层的维度列表，这些维度定义了生成器中残差层的大小。
        # data_dim：输出数据的维度，即生成数据的大小。
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        #input_dim：输入数据的维度。
        # discriminator_dim：判别器层的维度列表，这些维度定义了判别器中各层的大小。
        # pac（Pairwise Adversarial Conditioning）：这是一个特定的参数，用于将多个样本组合在一起进行判别，以增强判别器的稳定性。
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),## 获取生成器模型的参数
            lr=self._generator_lr,## 设置生成器的学习率
            betas=(0.5, 0.9),# # 设置Adam优化器的beta1和beta2参数
            weight_decay=self._generator_decay,# 设置生成器的权重衰减参数
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)#把处理完的训练数据分批次
        for i in epoch_iterator:#循环遍历每个epoch
            GAtotal_loss = 0  # 初始化总损失为0
            DAtotal_loss = 0  # 初始化总损失为0
            num_batches = 0  # 初始化批次计数器
            batch = []
            for id_ in range(steps_per_epoch):# 循环遍历每个epoch中的每个步骤

                # def check_gpu_availability():
                #     """检查是否有可用的GPU"""
                #     return torch.cuda.is_available()
                #
                # def get_gpu_id():
                #     """获取当前使用的GPU编号"""
                #     if check_gpu_availability():
                #         return torch.cuda.current_device()
                #     return None
                # # 检查是否能使用GPU
                # if check_gpu_availability():
                #     import pynvml
                #     import os
                #     gpu_id = get_gpu_id()
                #     gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                #     current_pid = os.getpid()
                #     print(f"当前使用的GPU编号: {gpu_id}")
                # else:
                #     print("当前没有可用的GPU，将使用CPU进行计算。")
                #     gpu_handle = None
                #     current_pid = None

                D_allstep=0
                for n in range(self._discriminator_steps):# 循环执行判别器的更新步骤。
                    #生成随机噪声 fakez。
                    fakez = torch.normal(mean=mean, std=std)
                    #如果 condvec 为 None，则不使用条件向量；否则，将条件向量与随机噪声拼接。
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )#从Train中随机抽取真实数据
                    else:
                        c1, m1, col, opt = condvec#将条件向量中的c1和m1转换为PyTorch张量，并移动到指定的设备
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)#将条件向量c1与随机噪声fakez拼接，以控制生成器的输出。

                        perm = np.arange(self._batch_size)#打乱索引perm，以便在真实数据采样时引入随机性。
                        np.random.shuffle(perm)

                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )

                        c2 = c1[perm]#根据打乱后的索引采样真实数据，并创建一个新的条件向量c2，它是c1的打乱版本。
                    #假数据
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    # 真数据
                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    #判别器的输出 y_fake 和 y_real
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)
                    #梯度惩罚 pen
                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    #W损失函数
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))+pen
                    # print(loss_d)
                    optimizerD.zero_grad(set_to_none=False)
                    # pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                    D_allstep+=loss_d

                #生成器训练
                fakez = torch.normal(mean=mean, std=std)#生成正太分布的噪声
                condvec = self._data_sampler.sample_condvec(self._batch_size)#从数据采样器中获得条件向量
                #如果 condvec 为 None，则不使用条件向量；否则，将条件向量与随机噪声拼接：
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:#如果没有条件向量，说明不需要计算离散特征的交叉熵损失。cross_entropy = 0：交叉熵损失为0。
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                # 每个批次的损失值
                GAbatch_loss = loss_g.detach().cpu().item()
                DAbatch_loss = D_allstep.detach().cpu().item()
                # print(f"Epoch {i}, Batch {id_}, Loss: {Tavebatch_loss}")
                num_batches += 1
                GAtotal_loss += GAbatch_loss
                DAtotal_loss += DAbatch_loss/self._discriminator_steps


                """为什么生成器对于每一个批次只需要循坏一次，而判别器需要循环多次
                判别器多次：
                这是因为：基于Wasserstein GAN的一个变种，其中判别器的更新次数通常大于或等于生成器的更新次数。
                这样做的目的是让判别器有更多的机会去适应生成器的变化，从而提高判别器的判别能力。
                生成器一次：
                这是因为生成器的目标是生成越来越真实的数据，而判别器的目标是区分真实数据和生成数据。
                如果生成器更新太频繁，可能会导致判别器无法正确学习。
                
                判别器有多次机会去学习如何区分真实数据和生成数据，而生成器则专注于生成更真实的数据。这种平衡对于训练稳定的GAN模型至关重要。
                通过这种方式，可以减少模式崩溃（mode collapse）的风险，即生成器生成的数据变得过于单一，失去了多样性。
                """
            GAaverage_loss = GAtotal_loss / num_batches
            DAaverage_loss = DAtotal_loss / num_batches
            # 打印 epoch 的平均损失
            print(f"Epoch {i + 1}, GAaverage Loss: {GAaverage_loss}, DAaverage Loss: {DAaverage_loss}")

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """n：需要采样的行数。
        condition_column：一个离散列的名称，用于条件采样。
        condition_value：在 condition_column 中我们希望增加其发生概率的类别名称。"""
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        #如果提供了 condition_column 和 condition_value，则通过 self._transformer 将列名和值转换为ID，
        # 然后使用这些ID生成一个全局条件向量 global_condition_vec，这个向量将用于增加特定条件的概率。
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None
        #计算需要多少次迭代来生成 n 行数据。
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)
            #如果有全局条件向量，则复制它；否则，从原始数据中采样一个条件向量。
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:#如果 condvec 不是 None，则将其转换为Tensor并附加到 fakez 上，以便在生成过程中考虑条件信息。
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
