"""TVAEGAN module."""

import numpy as np
import pandas as pd
import torch
from torch.nn import Embedding, Linear, Module, Parameter, ReLU,BatchNorm1d, LeakyReLU, Sequential, Conv1d, ConvTranspose1d, Flatten, Unflatten,functional

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from util.ctgan.data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#固定随机
g = torch.Generator()
g.manual_seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(Module):
    def __init__(self, feature_num, embedding_dim,num_class,conditional=False,use_bn=False):
        super(Encoder, self).__init__()
        self.conditional = conditional  # 新增条件控制标志
        self.use_bn = use_bn  # 控制是否启用BN层
        if self.conditional:
            # 输入是独热编码 [batch_size, num_class]，直接投影到条件特征
            self.cond_proj = nn.Linear(num_class, 32)  # 投影到32维
            conv1_in_channels = 1 + 32  # 原始输入通道 + 条件通道
        else:
            conv1_in_channels = 1

        self.conv1 = Conv1d(conv1_in_channels, 32, kernel_size=3, padding=0)#(32, feature_num-2)
        self.conv2 = Conv1d(32, 64, kernel_size=3, padding=0)#(64, feature_num-4)
        # 动态添加BN层
        if self.use_bn:
            self.bn1 = BatchNorm1d(32)
            self.bn2 = BatchNorm1d(64)
            self.bn_fc = BatchNorm1d(256)
        self.flatten = Flatten()# 64 * (feature_num - 4)
        self.fc1 = Linear(64 * (feature_num -4), 256)

        # 全连接层，计算潜在空间的均值，维度为128
        self.fc2 = Linear(256, embedding_dim)
        # 全连接层，计算潜在空间的方差，维度为128
        self.fc3 = Linear(256, embedding_dim)
        # self.attention = MultiHeadSelfAttention(channels=32)  # 初始化注意力机制

    def forward(self, input_,cond=None):# [batsh_size,1,特征数]
        if self.conditional and cond is not None:
            # 输入 cond 是独热编码 [batch_size, num_class]
            cond = cond.float()  # 关键修改
            cond_feat = F.relu(self.cond_proj(cond))  # [batch_size, 32]
            cond_feat = cond_feat.unsqueeze(-1)  # [batch_size, 32, 1]
            cond_feat = cond_feat.expand(-1, -1, input_.shape[-1])  # [batch_size, 32, seq_len]
            input_ = torch.cat([input_, cond_feat], dim=1)  # [batch_size, 1+32, seq_len]

            # x = ReLU()(self.conv1(input_))  # x([batsh_size,1,特征数])->[batsh_size,32,特征数-2]
        # 第一个卷积层，使用ReLU激活函数
        x = self.conv1(input_)## [batsh_size,32,特征数-2]
        if self.use_bn:
            x = self.bn1(x)
        x = ReLU()(x)
        # x = self.attention(x)  # 在第二个卷积层之后应用注意力机制
        # 第二个卷积层，使用ReLU激活函数
        x = self.conv2(x)#[batsh_size,64,特征数-4]
        if self.use_bn:
            x = self.bn2(x)
        x = ReLU()(x)
        # 扁平化操作
        x = self.flatten(x)#[batsh_size,64*特征数-4]
        # 全连接层，使用ReLU激活函数
        x = self.fc1(x)#[batsh_size,256]
        if self.use_bn:
            x = self.bn_fc(x)
        x = ReLU()(x)
        # 计算潜在空间的均值
        mu = self.fc2(x)#[batsh_size,128]
        # mu = self.tcn_1(mu.unsqueeze(1)).squeeze(1)
        # mu = mu[:, 0, :]
        # 计算潜在空间的对数方差
        logvar = self.fc3(x)
        # logvar = self.tcn_2(logvar.unsqueeze(1)).squeeze(1)
        # logvar = logvar[:, 0, :]
        # print(logvar.size())
        #std标准差
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=False):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # Adjust input shape if necessary
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return self.relu(x)

class Decoder(Module):
    def __init__(self, embedding_dim, feature_num, num_class, conditional=False,use_bn=False):
        super(Decoder, self).__init__()
        self.conditional = conditional  # 新增条件控制标志
        self.use_bn = use_bn  # 控制是否启用BN层

        if self.conditional:
            # 输入是独热编码 [batch_size, num_class]，直接投影到条件特征
            self.cond_proj = nn.Linear(num_class, 32)  # 投影到32维
            fc_in_dim = embedding_dim + 32  # 潜在变量维度 + 条件维度
        else:
            fc_in_dim = embedding_dim

        self.fc = Linear(fc_in_dim, 64 * (feature_num-4) )# 64 * (feature_num - 4)
        self.unflatten = Unflatten(dim=1, unflattened_size=(64, int(feature_num-4)))#转换成卷积三位结构 (64, feature_num - 4)
        # 第一个转置卷积层，用于上采样，输入通道为64，输出通道为64，卷积核大小为3，填充为1
        self.conv1 = ConvTranspose1d(64, 64, kernel_size=3, padding=0)# (64, feature_num - 2)
        # 第二个转置卷积层，用于上采样，输入通道为64，输出通道为32，卷积核大小为3，填充为1
        self.conv2 = ConvTranspose1d(64, 32, kernel_size=3, padding=0)#(32, feature_num)
        # 第三个转置卷积层，用于生成最终输出，输入通道为32，输出通道为1，卷积核大小为1
        self.conv3 = ConvTranspose1d(32, 1, kernel_size=1)#(1, feature_num)
        # 动态添加BN层
        if self.use_bn:
            self.bn_fc = BatchNorm1d(64 * (feature_num-4))
            self.bn1 = BatchNorm1d(64)
            self.bn2 = BatchNorm1d(32)
        # TCN 层
        # self.tcn = TemporalConvNet(1, [1,2])
        self.sigmoid= nn.Sigmoid()
        # self.attention = MultiHeadSelfAttention(64)
        self.sigma = Parameter(torch.ones(feature_num) * 0.1)

    def forward(self, input_, cond=None):  # cond参数改为可选
        if self.conditional and cond is not None:
            # 输入 cond 是独热编码 [batch_size, num_class]
            cond = cond.float()  # 关键修改
            cond_feat = F.relu(self.cond_proj(cond))  # [batch_size, 32]
            input_ = torch.cat([input_, cond_feat], dim=1)  # [batch_size, embedding_dim + 32]

        x = self.fc(input_)#[batch_size,64 * (feature_num-4)]
        if self.use_bn:
            x = self.bn_fc(x)
        x = ReLU()(x)
        # 调整形状，为转置卷积操作做准备
        x = self.unflatten(x)#[batch_size,64 , (feature_num-4)]
        # 第一个转置卷积层，使用ReLU激活函数Q
        x = self.conv1(x)#[batch_size,32 , (feature_num-2)]
        if self.use_bn:
            x = self.bn1(x)
        x = ReLU()(x)
        # x = self.attention(x)

        # 第二个转置卷积层，使用ReLU激活函数
        x = self.conv2(x)#[batch_size,32 , feature_num]
        if self.use_bn:
            x = self.bn2(x)
        # x = self.attention(x
        x = ReLU()(x)
        # 第三个转置卷积层，不使用激活函数
        x = self.conv3(x)#[batch_size,1 , feature_num]
        return x, self.sigma
""""判别器"""
class Discriminator(Module):
    def __init__(self, feature_num,num_class,use_bn=False):
        super(Discriminator, self).__init__()
        self.use_bn = use_bn  # 控制是否启用BN层
        # self.embedding = Embedding(num_class, feature_num)  # 嵌入维度可以根据需要调整
        # 第一个卷积层，输入通道为特征数，输出通道为32，卷积核大小为3，填充为1
        self.conv1 = Conv1d(1, 32, kernel_size=3, padding=0)# (32, feature_num - 2)
        # 第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3，填充为1
        self.conv2 = Conv1d(32, 64, kernel_size=3, padding=0)#(64, feature_num - 4)
        self.flatten = Flatten()#64 * (feature_num - 4)
        # 全连接层，将卷积层输出的特征映射到512维
        self.fc1 = Linear(64 * (feature_num-4), 256)
        # self.fc1 = Linear(64 * (self.pac - 2), 512)
        # 全连接层，用于真实性判断，输出维度为1
        self.fc2 = Linear(256, 1)
        self.softmax = nn.Softmax()
        # 全连接层，用于类别预测，输出维度为类别数[ACGAN的]
        # 动态添加BN层
        if self.use_bn:
            self.bn1 = BatchNorm1d(32)
            self.bn2 = BatchNorm1d(64)
            self.bn_fc = BatchNorm1d(256)
        # 新增分类分支
        # 分类头（关键修改）
        self.classifier = Sequential(
            Linear(256, 64),
            Linear(64, num_class),
            nn.Softmax(dim=1)
        )
        # 真实性判别头
        self.validity_head = Sequential(
            Linear(256, 64),
            Linear(64, 1),
        )

    def forward(self, input_):  # [batsh_size,1,特征数]
        # 特征提取
        x = self.conv1(input_)
        if self.use_bn: x = self.bn1(x)
        x = ReLU()(x)

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = ReLU()(x)

        x = self.flatten(x)
        x = self.fc1(x)
        if self.use_bn: x = self.bn_fc(x)
        features = ReLU()(x)

        # 多任务输出
        validity = self.validity_head(features)  # 真实性评分
        class_probs = self.classifier(features)  # 类别概率
        # 类别概率输出
        return validity,features,class_probs

    #梯度惩罚计算  WGAN中
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu',  lambda_=10):
        """self不同 alpha有pac gradient_view有点区别"""
        #alpha是一个随机张量，用于在真实数据和生成数据之间插值。
        # alpha = torch.rand(real_data.size(0) // pac, 1, 1,device=device)  # pac 确保 alpha 的第一个维度是批次大小除以 pac，这意味着每个 alpha 将用于 pac 个样本
        # alpha = alpha.repeat(1, pac, real_data.size(1))
        # alpha = alpha.view(-1, real_data.size(1))
        alpha = torch.rand(real_data.size(0), *real_data.size()[1:], device=device)#alpha[batch_size,1，特征数]
        #interpolates是插值后的数据。这个插值过程可以看作是在真实数据和生成数据之间的直线上随机采样点
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)#interpolates[batch_size,1，特征数]
        #disc_interpolates是判别器对插值数据的输出。 用于计算梯度
        disc_interpolates,_,_ = self(interpolates)#disc_interpolates[batch_size,1]
        #gradients是插值数据相对于判别器输出的梯度。
        gradients = torch.autograd.grad(#gradients[batch_size,1，特征数]
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        #gradients_view是梯度的范数减1。
        gradients_view = gradients.view(-1,  real_data.size(1)).norm(2, dim=1) - 1
        #gradient_penalty是梯度惩罚项，用于正则化判别器。
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_#gradient_penalty变成了一个数
        return gradient_penalty


class PRECVGMTVAEGANConv1d(BaseSynthesizer):
    def __init__(
        self,
        embedding_dim=128,
        discriminator_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        loss_factor=2,
        cuda=True,
        verbose=False,
        c=0.005,
        discriminator_train_steps=5,
        dlr=0.0001,
        pltpath=None,
        lr1=0.001,
        lr2=0.001,
        num_class=4,
    ):
        self.embedding_dim = embedding_dim#潜在空间的维度
        self.compress_dims = compress_dims#编码其中隐藏层的维度列表
        self.decompress_dims = decompress_dims#解码器中隐藏层的维度列表
        self.discriminator_dim=discriminator_dim
        self.l2scale = l2scale#L2正则化系数
        self.loss_factor = loss_factor#损失因子
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose#是否在训练过程中显示进度条和损失信息
        self.c=c
        self.discriminator_train_steps=discriminator_train_steps
        self.dlr = dlr
        self.pltpath=pltpath
        self.num_class=num_class

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self.decoder=None
        self.encoder=None
        self.classfy=None
        self._transformer = None
        self.loss_values = None
        self.adaptive_fc = None
        self.init_params = None
        self.fisher = None
        self.lr1 = lr1
        self.lr2 = lr2
        self.conditional=False
        self.use_bn=False
        # ===== 新增：类别条件先验参数 =====
        self.num_class = self.num_class
        self.class_mean_embeddings = nn.Embedding(self.num_class, self.embedding_dim).to(self._device)
        self.class_logvar_embeddings = nn.Embedding(self.num_class, self.embedding_dim).to(self._device)


        # 初始化参数
        nn.init.normal_(self.class_mean_embeddings.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.class_logvar_embeddings.weight, val=0.0)

    # 在训练循环中修改KL计算
    def _kl_divergence(self, mu, logvar, label):
        label = label.to(self._device)
        label_index = label.argmax(dim=1).long().to(self._device)

        class_prior_mean = self.class_mean_embeddings(label_index)
        class_prior_logvar = self.class_logvar_embeddings(label_index)

        kl = 0.5 * torch.sum(
            class_prior_logvar - logvar
            + (logvar.exp() + (mu - class_prior_mean).pow(2)) / class_prior_logvar.exp()
            - 1
        )
        return kl
    @random_state
    def fit(self, train_data,label,discrete_columns,batch_size,epoch,pretrain=True, finetune=False, pretrainpath=None,savepath=None,pltpath=None,dlr=0.001,lr1=0.0001,lr2=0.0001):
        print("开始数据预处理...")
        self._transformer = DataTransformer()
        print('1')
        self._transformer.fit(train_data, discrete_columns)
        print('2')
        train_data = self._transformer.transform(train_data)

        # 创建 TensorDataset 和 DataLoader 用于批量加载数据。
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device),torch.from_numpy(label.values).float().to(self._device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,generator=g)
        # data_dim = train_data.shape[1]+label.shape[1]
        # features_dim=train_data.shape[1]
        features_dim=self._transformer.output_dimensions
        print(f"特征维度{features_dim},标签维度{label.shape[1]}")

        #预训练
        if pretrain:

            print("初始化编码器和解码器...")
            # 初始化编码器和解码器，并将其移动到指定的设备（GPU或CPU）。
            conditional = True  # 新增条件模式标志
            self.conditional=conditional  # 新增条件模式标志
            use_bn = False  # 新增条件模式标志
            self.use_bn = use_bn  # 新增条件模式标志
            self.encoder = Encoder(features_dim, self.embedding_dim,self.num_class,conditional=conditional,use_bn=use_bn).to(self._device)
            self.decoder = Decoder(self.embedding_dim,features_dim,self.num_class,conditional=conditional,use_bn=use_bn).to(self._device)
            print("设置解码器和编码器优化器...")
            # optimizerAE = Adam(
            #     list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
            # )
            self.optimizer_encoder = Adam(self.encoder.parameters(), weight_decay=self.l2scale,  betas=(0.5, 0.9), lr=lr1 )
            self.optimizer_decoder = Adam(self.decoder.parameters(), weight_decay=self.l2scale,  betas=(0.5, 0.9), lr=lr2)
            print("初始化判别器...")
            self.discriminator = Discriminator(features_dim,self.num_class,use_bn=use_bn).to(self._device)
            self.optimizerD = Adam(self.discriminator.parameters(), weight_decay=self.l2scale, lr=dlr)
            # self.classfy = Discriminator(features_dim,self.num_class).to(self._device)
            print("设置判别器优化器...")
            # self.optimizerD = Adam( list(self.discriminator.parameters()) + list(self.classfy.parameters()), weight_decay=self.l2scale,  betas=(0.5, 0.9), lr=self.dlr)


            #添加学习率调度器，动态调整学习率
            # 在初始化优化器后添加
            # self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(self.optimizer_encoder, step_size=50, gamma=0.5)
            # self.scheduler_decoder = torch.optim.lr_scheduler.StepLR(self.optimizer_decoder, step_size=50, gamma=0.5)
            # self.schedulerD = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=50, gamma=0.5)



            print("开始预训练...")
            self.train(loader,batch_size,epoch,pltpath,finetune=False,pretrain=True)


            # 保存预训练模型
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),

            }, savepath)
            print(f"预训练模型已保存到 {savepath}")

        if finetune:
            if pretrainpath is None:
                raise ValueError("进行微调时，必须提供预训练模型的路径。")
            print(f"从 {pretrainpath} 加载预训练模型...")

            checkpoint = torch.load(pretrainpath)
            # 获取预训练模型的特征维度
            pretrain_fc1_weight_shape = checkpoint['encoder_state_dict']['fc1.weight'].shape[1]
            target_dim = (pretrain_fc1_weight_shape // 64) + 4

            # 如果当前数据维度与预训练维度不同，添加自适应全连接层
            if features_dim != target_dim:
                self.adaptive_fc = Linear(features_dim, target_dim).to(self._device)
                self.optimizer_adaptive_fc = Adam(self.adaptive_fc.parameters(), weight_decay=self.l2scale, lr=0.001)



            conditional = True  # 新增条件模式标志
            self.conditional = conditional  # 新增条件模式标志
            use_bn = True  # 新增条件模式标志
            self.use_bn = use_bn  # 新增条件模式标志
            self.encoder = Encoder(features_dim, self.embedding_dim, self.num_class, conditional=conditional,use_bn=use_bn).to(self._device)
            self.decoder = Decoder(self.embedding_dim, features_dim, self.num_class, conditional=conditional,use_bn=use_bn).to(self._device)
            self.discriminator = Discriminator(features_dim, self.num_class,use_bn=use_bn).to(self._device)



            # 加载 encoder 状态字典，跳过形状不匹配的参数
            encoder_state_dict = self.encoder.state_dict()
            pretrained_encoder_state_dict = checkpoint['encoder_state_dict']
            for name, param in pretrained_encoder_state_dict.items():
                if name in encoder_state_dict and encoder_state_dict[name].shape == param.shape:
                    encoder_state_dict[name] = param
            self.encoder.load_state_dict(encoder_state_dict)

            # 加载 decoder 状态字典，跳过形状不匹配的参数
            decoder_state_dict = self.decoder.state_dict()
            pretrained_decoder_state_dict = checkpoint['decoder_state_dict']
            for name, param in pretrained_decoder_state_dict.items():
                if name in decoder_state_dict and decoder_state_dict[name].shape == param.shape:
                    decoder_state_dict[name] = param
            self.decoder.load_state_dict(decoder_state_dict)

            # 加载 discriminator 状态字典，跳过形状不匹配的参数
            discriminator_state_dict = self.discriminator.state_dict()
            pretrained_discriminator_state_dict = checkpoint['discriminator_state_dict']
            for name, param in pretrained_discriminator_state_dict.items():
                if name in discriminator_state_dict and discriminator_state_dict[name].shape == param.shape:
                    discriminator_state_dict[name] = param
            self.discriminator.load_state_dict(discriminator_state_dict)



            # 冻结编码器的参数
            # 例如，冻结编码器的前两个卷积层
            # for param in self.encoder.conv1.parameters():
            #     param.requires_grad = False
            # for param in self.encoder.conv2.parameters():
            #     param.requires_grad = False
            #
            # self.optimizer_encoder = Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()),
            #                               weight_decay=self.l2scale, lr=self.lr1)
            self.optimizer_encoder = Adam(self.encoder.parameters(), weight_decay=self.l2scale, lr= lr1)
            self.optimizer_decoder = Adam(self.decoder.parameters(), weight_decay=self.l2scale, lr= lr2)
            self.optimizerD = Adam(self.discriminator.parameters(), weight_decay=self.l2scale, lr=dlr)
            print("开始微调...")
            self.train(loader,batch_size,epoch,pltpath,finetune=True,pretrain=False)

    def train(self, loader, batch_size, epoch, pltpath, pretrain=True, finetune=False):
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'EnLoss', 'DeLoss', 'DLoss'])
        clip_value = 1.0

        iterator = tqdm(range(epoch), disable=(not self.verbose))
        if self.verbose:
            iterator.set_description('Loss: {loss:.3f}'.format(loss=0))

        for i in iterator:
            print(f"开始第 {i + 1} 个 epoch 的训练...")
            En_epoch_loss = 0
            De_epoch_loss = 0
            D_epoch_loss = 0
            num_batches = 0

            for batch_idx, (feature, label) in enumerate(loader):
                # ================== 判别器训练 ==================
                D_batch_loss = 0.0
                for _ in range(self.discriminator_train_steps):
                    self.discriminator.zero_grad()

                    real = feature.unsqueeze(1).to(self._device)
                    real_label = label.long().to(self._device)

                    # 真实数据前向
                    real_out, _, real_cls = self.discriminator(real)
                    real_cls_loss = F.cross_entropy(real_cls, real_label.float())

                    # 生成数据（禁用生成器梯度）
                    # with torch.no_grad():
                    mu, std, _ = self.encoder(real, real_label)
                    emb = mu + torch.randn_like(std) * std
                    rec_data, _ = self.decoder(emb, real_label)

                    # 假数据前向
                    rec_out, _, fake_cls = self.discriminator(rec_data)
                    fake_cls_loss = F.cross_entropy(fake_cls, real_label.float())

                    # 损失计算
                    cls_loss = (real_cls_loss + fake_cls_loss)*0.1
                    adv_loss = -torch.mean(real_out) + torch.mean(rec_out)
                    gp = self.discriminator.calc_gradient_penalty(real, rec_data, self._device)
                    d_loss = adv_loss + gp + cls_loss
                    # d_loss = adv_loss + gp #消融无分类损失
                    # 反向传播
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_value)
                    self.optimizerD.step()

                    D_batch_loss += d_loss.item()

                # 计算当前batch的判别器平均损失
                D_batch_avg = D_batch_loss / self.discriminator_train_steps
                D_epoch_loss += D_batch_avg

                # ================== 生成器训练 ==================
                self.encoder.zero_grad()
                self.decoder.zero_grad()

                mu, std, logvar = self.encoder(real, real_label)
                emb = mu + torch.randn_like(std) * std
                rec, _ = self.decoder(emb, real_label)

                # 判别器前向
                fake_out, real_feat, fake_cls = self.discriminator(rec)
                _, rec_feat, _ = self.discriminator(rec)

                # 损失计算
                KLD = self._kl_divergence(mu, logvar, real_label)
                cls_loss = F.cross_entropy(fake_cls, real_label.float())*0.1

                #对比损失
                feat_loss = F.mse_loss(real_feat, rec_feat)
                # _, real_intermediate,_ = self.discriminator(real)
                # rec_out, rec_intermediate,_ = self.discriminator(rec)
                # mean = rec_intermediate  # 假设均值为中间特征#mean=[batch_size,512]
                # var = torch.ones_like(rec_intermediate)  # 假设方差全为1[batch_size,512]
                # dist = torch.distributions.Normal(mean, torch.sqrt(var))  # [batch_size,512]
                # log_prob = dist.log_prob(real_intermediate)  # [batch_size,512]
                # mloss = -torch.mean(log_prob)  # 一个浮点数

                adv_g_loss = -torch.mean(fake_out)


                en_loss = KLD + feat_loss
                de_loss = adv_g_loss + cls_loss + feat_loss
                # de_loss =  cls_loss + feat_loss
                # print(adv_g_loss)
                # print(cls_loss)

                # de_loss = adv_g_loss  + feat_loss#无分类损失

                # 反向传播
                en_loss.backward(retain_graph=True)
                de_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_value)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip_value)
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()

                # 记录损失
                En_epoch_loss += en_loss.item()
                De_epoch_loss += de_loss.item()
                num_batches += 1

                # 将当前batch的损失记录到DataFrame
                batch_loss_df = pd.DataFrame({
                    'Epoch': [i],
                    'Batch': [batch_idx],
                    'EnLoss': [en_loss.item()],
                    'DeLoss': [de_loss.item()],
                    'DLoss': [D_batch_avg]
                })
                self.loss_values = pd.concat([self.loss_values, batch_loss_df], ignore_index=True)

            # 计算epoch平均损失
            avg_en = En_epoch_loss / num_batches
            avg_de = De_epoch_loss / num_batches
            avg_d = D_epoch_loss / num_batches
            print(f"Epoch {i + 1} | En: {avg_en:.2f} | De: {avg_de:.2f} | D: {avg_d:.2f}")


    def plot_feature_impacts(self, train_data, feature_impacts, feature_names=None):
        """
        绘制特征影响值图。

        参数:
        feature_impacts (numpy.ndarray): 特征影响值数组。
        feature_names (list, optional): 特征名称列表。如果提供，将用于x轴标签。
        """
        plt.figure(figsize=(12, 6))

        if feature_names is not None:
            if len(feature_names) != len(feature_impacts):
                raise ValueError("Length of feature_names must match length of feature_impacts")
            plt.xticks(range(len(feature_impacts)), feature_names, rotation=90)
        else:
            plt.xticks(range(len(feature_impacts)), rotation=90)
        plt.bar(range(len(feature_impacts)), feature_impacts)
        plt.xlabel("Feature Index" if feature_names is None else "Feature Name")
        plt.ylabel("Feature Impact")
        plt.title("Feature Impacts")
        plt.tight_layout()
        plt.show()
    def plot_losses(self,pltpath):
        # 计算每个 epoch 的平均损失
        epoch_losses = self.loss_values.groupby('Epoch').mean()

        plt.figure(figsize=(25, 20))

        # 绘制 VAE 损失
        plt.subplot(1, 3, 1)
        plt.plot(epoch_losses['EnLoss'], label='EN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Encode Loss per Epoch')
        plt.legend()
        # 绘制 VAE 损失
        plt.subplot(1, 3, 2)
        plt.plot(epoch_losses['DeLoss'], label='DE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Decode Loss per Epoch')
        plt.legend()

        # 绘制判别器损失
        plt.subplot(1, 3, 3)
        plt.plot(epoch_losses['DAloss'], label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss per Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(pltpath, dpi=300)
        # plt.show()

    @random_state
    def sample(self, samples, batch_size, majority_class_data, num_class, cond):
        self.decoder.eval()
        self.encoder.eval()
        self.discriminator.eval()
        quality_threshold=0.5

        steps = samples // batch_size + 1
        data = []

        if majority_class_data is None:
            raise ValueError("多数类数据未赋值，请先设置 self.majority_class_data")

        # majority_class_data_tensor = torch.tensor(majority_class_data.values, dtype=torch.float32).to(self._device)
        # random_indices = torch.randperm(majority_class_data_tensor.shape[0])[:batch_size]
        # sampled_majority_data = majority_class_data_tensor[random_indices]
        # 生成形状为[B, 5]的条件向量，类别为3
        condition_vector = torch.zeros(batch_size, num_class).to(self._device)
        condition_vector[:, cond] = 1
        condition_vector = condition_vector.long()

        for i in range(steps):
            mean = torch.zeros(batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            # with torch.no_grad():
            #     mu, std, logvar = self.encoder(sampled_majority_data.unsqueeze(1))
            # eps = torch.randn_like(std)  # 生成随机噪声，并将其与均值和标准差结合，生成潜在向量。
            # emb = eps * std + mu  # 通过解码器重建数据。

            fake, sigmas = self.decoder(noise, condition_vector)
            validity, _, cls_prob =  self.discriminator(fake)  # 判别器输出置信度
            # print(cls_prob.argmax(1) )
            # print(validity.sigmoid() > quality_threshold)
            # mask = (validity.sigmoid() > quality_threshold) & (cls_prob.argmax(1) == cond)
            fake = torch.sigmoid(fake.squeeze(1))
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]

        return self._transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
    #设置设备
    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
        if self.adaptive_fc:
            self.adaptive_fc.to(self._device)

    #分类器
    def predict(self, X):
        """分类预测方法"""
        self.discriminator.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self._device)
            _,_, logits = self.discriminator(X_tensor)
            # 找到每行最大概率的索引
            pred_classes = np.argmax(logits.cpu().numpy(), axis=1)
            return pred_classes