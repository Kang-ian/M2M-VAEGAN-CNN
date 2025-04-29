
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, \
    GlobalAveragePooling1D, Flatten
from sklearn import metrics
# from keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
from util.ctgan.synthesizers.PreCVGMtvaeganConv1d import PRECVGMTVAEGANConv1d

import torch
import numpy as np
import tensorflow as tf
import random as python_random
from keras.initializers import glorot_uniform
import glob
# 固定所有随机种子
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# 设置 TensorFlow 操作的确定性
tf.config.experimental.enable_op_determinism()

# 固定 torch 随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # 如果使用多个 GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------加载数据
path = "data/cic/preprocessed/DATA.csv"
if not os.path.exists(path):
    directory_path = 'data/ids/MachineLearningCVE/'  # CICIDS2017
    csv_files = glob.glob(directory_path + '*.csv')  # 使用 glob.glob 获取所有 CSV 文件的路径列表。
    dataframes = []
    for file in csv_files:  # 遍历每个文件，使用 pd.read_csv 读取数据，并将其添加到 dataframes 列表中。
        dataframe = pd.read_csv(file)
        dataframes.append(dataframe)
    data = pd.concat(dataframes, ignore_index=True)  # 使用 pd.concat 将所有数据框合并成一个大的数据框 data，
    print(data[' Label'].value_counts())
    data = data.sample(frac=0.3, random_state=42)  # 随机取百分十之十的数据，并把有些样本数太少的删掉
    # print(data[' Label'].value_counts())
    # 要删除的标签列表
    labels_to_drop = [
        'Bot',
        'Web Attack � Brute Force',
        'Web Attack � XSS',
        'Infiltration',
        'Heartbleed',
        'Web Attack � Sql Injection'
    ]
    # 删除指定的行
    # 删除指定的行
    data = data[~data[' Label'].isin(labels_to_drop)]  # 使用 isin 方法过滤掉指定的标签。
    print(data[' Label'].value_counts())
    data.to_csv(path, index=False)  # 将处理后的数据框 data 保存为 gzip 压缩的 CSV 文件。
else:
    data = pd.read_csv(path)
data.columns = data.columns.str.strip()  # 去除列名中的空白。
data = data.dropna()  # 移除数据中的空值（data.dropna()）。


# 2、多分类编码
label_mapping = {
    'BENIGN': 0,
    'DoS Hulk': 1,
    'PortScan': 2,
    'DDoS': 3,
    'DoS GoldenEye': 4,
    'FTP-Patator': 5,
    'DoS slowloris': 6,
    'SSH-Patator': 7,
    'DoS Slowhttptest': 8
}
# 对'Label'列进行编码
data['Label'] = data['Label'].map(label_mapping)
print(f"原始数据分布{data['Label'].value_counts()}")


# 处理异常值
if np.isinf(data).values.any() or np.isnan(data).values.any():
    print("数据中存在无穷大值或NaN值，正在删除包含这些值的样本...")

    # 先用 replace() 将 inf 替换为 NaN，然后用 dropna() 删除所有异常值
    data_cleaned = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 重新提取特征和标签
    features = data_cleaned.drop(columns=['Label'])
    label = data_cleaned['Label']

    print(f"删除后的数据形状: {data_cleaned.shape}")
else:
    print("数据中没有无穷大值或NaN值，无需删除样本。")


train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
print(f"切割训练集数据分布{train_data['Label'].value_counts()}")
print(f"切割测试集集数据分布{test_data['Label'].value_counts()}")


#划分类
majority_classes = [1,2,3]
minority_classes=[4,5,6,7,8]
majority_train_data = train_data[train_data['Label'].isin(majority_classes)]
minority_train_data = train_data[train_data['Label'].isin(minority_classes)]
# 从每一类中直接选取 1000 个样本
sample_size = 4000
balanced_samples = (
    majority_train_data
    .groupby('Label', group_keys=False)
    .apply(lambda x: x.sample(n=sample_size, random_state=0))
)
majority_train_data=balanced_samples
print(f"新majority_train_data数据分布: {balanced_samples['Label'].value_counts()}")


# 替换 inf 为 NaN，然后删除所有 NaN 值
majority_train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
majority_train_data.dropna(inplace=True)

# 检查是否仍有 inf 或 NaN
if np.isinf(minority_train_data.values).any() or np.isnan(minority_train_data.values).any():
    raise ValueError("数据仍然包含 inf 或 NaN，请检查数据预处理！")

# 替换 inf 为 NaN，然后删除所有 NaN 值
minority_train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
minority_train_data.dropna(inplace=True)

# 检查是否仍有 inf 或 NaN
if np.isinf(minority_train_data.values).any() or np.isnan(minority_train_data.values).any():
    raise ValueError("数据仍然包含 inf 或 NaN，请检查数据预处理！")

#---------------------------------VAEGAN
from GPUtil import getGPUs
import time
#记录GPU和时间
def log_gpu_usage():
    if torch.cuda.is_available():
        gpu = getGPUs()[0]
        return {
            'used_mb': gpu.memoryUsed,
            'total_mb': gpu.memoryTotal
        }
    return {'used_mb': 0, 'total_mb': 0}

# 在预训练代码段修改如下：
print("开始预训练...")
start_time = time.time()
peak_mem = 0
all_classes = [1, 2, 3, 4,5,6,7,8]

#PREGMVAEGAN
# #Pretvaegan
loadpath1='data/cic/PRECVAEGANConv1d-GM条件0-majority-无0类.pth'
loadpath2='data/cic/PRECVAEGANConv1d-GM条件0-minority-无0类.pth'
random_seed = 42
discrete_columns=[]
synthesizer= PRECVGMTVAEGANConv1d(
    embedding_dim=200,
    l2scale=1e-5,
    loss_factor=2,
    cuda=True,
    verbose=False,
    c=0.005,
    discriminator_train_steps=5,
    num_class=len(all_classes)
)
# 预训练
synthesizer.set_random_state(random_seed)
# majority_train_data = majority_train_data.sample(n=10000)  # n 是抽取的样本数量
print(f"majority_train_data分布：{majority_train_data['Label'].value_counts()}")
majority_subset = majority_train_data.sample(frac=0.3, random_state=random_seed)
print(f"majority_subset分布：{majority_subset['Label'].value_counts()}")
tmp=majority_train_data.pop('Label')
majority_label=pd.get_dummies(tmp).astype(int)
# 确保独热编码后的列顺序与所有可能的类别顺序一致

for cls in all_classes:
    if cls not in majority_label.columns:
        majority_label[cls] = 0
majority_label = majority_label[all_classes]

synthesizer.fit(majority_train_data,majority_label,discrete_columns, batch_size=300,epoch=500, pretrain=True, finetune=False,savepath=loadpath1,pltpath='data/cic/synthesizer-majority',  dlr=0.00001,   lr1=0.0001,  lr2=0.0001)

# 记录预训练性能
pretrain_time = time.time() - start_time
pretrain_mem = log_gpu_usage()['used_mb']
print(f"预训练耗时: {pretrain_time:.2f}s | 峰值内存: {pretrain_mem}MB")

#----------------------------------------------------------------------------------
print('微调')
# 少数类微调数据
mixed_data=pd.concat([majority_subset,minority_train_data],ignore_index=True)
print(f"mixed_data分布：{mixed_data['Label'].value_counts()}")

# 在微调代码段修改如下：
start_time = time.time()
# 从多数类中随机抽取20%的数据xuAN
tmp=mixed_data.pop('Label')
mixed_label=pd.get_dummies(tmp).astype(int)
for cls in all_classes:
    if cls not in mixed_label.columns:
        mixed_label[cls] = 0
mixed_label = mixed_label[all_classes]


if os.path.exists(loadpath2):
    synthesizer = PRECTVAEGANConv1d.load(loadpath2)
else:
    synthesizer.set_random_state(random_seed)
    synthesizer.fit(mixed_data, mixed_label,discrete_columns,pretrain=False,batch_size=300,epoch=250, finetune=True,pretrainpath=loadpath1, savepath=loadpath2,pltpath='data/KDD/pre-synthesizer_R2L',dlr=0.00001,lr1=0.0001,lr2=0.0001)
    synthesizer.save(loadpath2)
#采样
target_labels=[4,5,6,7,8]
syndata=pd.DataFrame()
for i in target_labels:
    synth_data_i = synthesizer.sample(15000, batch_size=300, majority_class_data=mixed_data,
                                        num_class=len(all_classes), cond=i-1)
    synth_data_i['Label'] = i
    syndata = pd.concat([syndata, synth_data_i], ignore_index=True)


combined_data_2 = pd.concat([train_data,syndata],ignore_index=True)
print(f"合成训练数据分布: {combined_data_2['Label'].value_counts()}")

# 去重操作
unique_data = combined_data_2.drop_duplicates()
print(f"最终训练数据分布: {unique_data['Label'].value_counts()}")

# train_df_2= unique_data
# test_df_2=test_data
# print(f"训练集分布：{train_df_2['Label'].value_counts()}")
# print(f"测试集分布：{test_df_2['Label'].value_counts()}")
combined_data_3 = pd.concat([unique_data,test_data],ignore_index=True)

#--------------------------归一化
# 训练集
features = combined_data_3.drop('Label', axis=1)
label = combined_data_3['Label']
max_float64 = np.finfo(np.float64).max  # 将所有特征列中超出 np.float64 最大值的数值替换为这个最大值。
features = features.where(features <= max_float64, max_float64)

from sklearn.preprocessing import  MinMaxScaler
scalertrain = MinMaxScaler()  # 特征缩放,将所有特征缩放到 [0, 1] 区间内
scaled_data = scalertrain.fit_transform(features)
combined_data_3 = pd.DataFrame(data=scaled_data, index=features.index, columns=features.columns)
combined_data_3 = pd.concat([combined_data_3, label], axis=1)

train_size = len(unique_data)
test_size = len(test_data)
# 拆分数据集
train_df_2 = combined_data_3.iloc[:train_size]
test_df_2 = combined_data_3.iloc[train_size:train_size + test_size]
print(f"训练集分布：{train_df_2['Label'].value_counts()}")
print(f"测试集分布：{test_df_2['Label'].value_counts()}")
#
# #划分数据集
y_train = train_df_2['Label']
y_test = test_df_2['Label']
X_train = train_df_2.drop('Label',  axis=1)
X_test = test_df_2.drop('Label',  axis=1)



#建立模型
cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(78, 1)))
cnn.add(Convolution1D(64, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(Convolution1D(128, 3, padding="same", activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu", kernel_initializer=glorot_uniform(seed=42)))
cnn.add(Dropout(0.1))                    #0.2
cnn.add(Dense(128, activation="relu", kernel_initializer=glorot_uniform(seed=42)))
cnn.add(Dropout(0.1))
cnn.add(Dense(64, activation="relu", kernel_initializer=glorot_uniform(seed=42)))
 #added
cnn.add(Dropout(0.1))
cnn.add(Dense(64, activation="relu", kernel_initializer=glorot_uniform(seed=42)))
#added 2
cnn.add(Dropout(0.1))
cnn.add(Dense(9, activation="softmax", kernel_initializer=glorot_uniform(seed=42)))


cnn.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])

#进一步将训练集拆分成训练和验证集   ?是否有必要
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)#101随机种子

# from sklearn.model_selection import KFold
# # 定义10折交叉验证
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
#又最大最小?
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# X_train = MinMaxScaler().fit_transform(X_train)
# X_test = MinMaxScaler().fit_transform(X_test)

#训练集: 据从二维数组（[样本数, 特征数]）重塑为三维数组（[样本数, 特征数, 1]）
x_columns_train = train_df_2.columns.drop('Label')
x_train_array = train_X[x_columns_train].values
x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
# x_train_1 = np.array(x_train_array)

dummies = pd.get_dummies(train_y)  #将分类标签（train_y 和 y_test）转换为独热编码（One-Hot Encoding）形式。
outcomes = dummies.columns
num_classes = len(outcomes)
y_train_1 = dummies.values#分类标签独热编码后的值

#验证集
x_columns_test = test_df_2.columns.drop('Label')
x_valid_array = test_X[x_columns_test].values
x_test_1 = np.reshape(x_valid_array, (x_valid_array.shape[0], x_valid_array.shape[1], 1))
# x_test_1 = np.array(x_valid_array)

dummies_test = pd.get_dummies(test_y)  # Classification
outcomes_test = dummies_test.columns
num_classes = len(outcomes_test)
y_test_1 = dummies_test.values

#测试集
x_columns_test = test_df_2.columns.drop('Label')
x_test_array = test_df_2[x_columns_test].values
x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))
# x_test_2 = np.array(x_test_array)

dummies_test = pd.get_dummies(y_test)  # Classification
outcomes_test = dummies_test.columns
# num_classes = len(outcomes_test)
y_test_2 = dummies_test.values

history= cnn.fit(x_train_1, y_train_1,validation_data=(x_test_1,y_test_1), epochs=30,batch_size=128)

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#进行测试集预测
pred1 = cnn.predict(x_test_2)
pred = np.argmax(pred1,axis=1)#将概率矩阵转换为预测的类别标签。np.argmax 会返回每行中最大值的索引，即预测的类别。
# pred = le.fit_transform(pred2)#使用 LabelEncoder 对预测的类别标签进行编码。然而，这里的 fit_transform 是多余的，因为 pred2 已经是整数形式的类别标签。
#pred = le.inverse_transform(pred)
y_eval = np.argmax(y_test_2,axis=1)#将测试集的真实标签（独热编码形式）转换为整数形式的类别标签
score = metrics.accuracy_score(y_eval, pred)
print("Validation score: {:.4f}%".format(score*100))

acc = accuracy_score(y_eval, pred)
print("accuracy :", acc)
recall = recall_score(y_eval, pred, average=None)
print("recall : ", [round(x, 4) for x in recall])
precision = precision_score(y_eval, pred, average=None)
print("precision : ", [round(x, 4) for x in precision])
f1_scr = f1_score(y_eval, pred, average=None)
print("f1_score : ", [round(x, 4) for x in f1_scr])

# 0:Dos  1:normal  2:Probe  3:R2L  4:U2L
print("####   0:normal  1:Dos  2:Probe  3:R2L  4:U2L  ###\n\n")
print(classification_report(y_eval, pred,digits=4))

cm = confusion_matrix(y_eval, pred)
print(np.round(cm, 4),'\n')
fig, ax = plt.subplots(figsize=(8, 8))

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

print("TP = ", [round(x, 4) for x in TP])
print("TN = ", [round(x, 4) for x in TN])
print("FP = ", [round(x, 4) for x in FP])
print("FN = ", [round(x, 4) for x in FN])
print("\n")

# 计算 DR、TPR、G_means 的 macro avg 和 weighted avg
support = np.bincount(y_eval)
total_samples = len(y_eval)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("TPR = ",  [round(x, 4) for x in TPR], "  True positive rate, Sensitivity, hit rate, or recall")
tpr_macro_avg = np.mean(TPR)
tpr_weighted_avg = np.sum(TPR * support) / total_samples
print("TPR macro avg: {:.4f}".format(tpr_macro_avg))
print("TPR weighted avg: {:.4f}".format(tpr_weighted_avg))

#DR
DR = TP / (TP + FP)
print("DR = ",  [round(x, 4) for x in DR])
dr_macro_avg = np.mean(DR)
dr_weighted_avg = np.sum(DR * support) / total_samples
print("DR macro avg: {:.4f}".format(dr_macro_avg))
print("DR weighted avg: {:.4f}".format(dr_weighted_avg))

TNR = TN / (TN + FP)
print("TNR = ",  [round(x, 4) for x in TNR], "  True negative rate or specificity")

G_means = np.sqrt(TPR * TNR)
print("G_means = ", [round(x, 4) for x in G_means])
g_means_macro_avg = np.mean(G_means)\




g_means_weighted_avg = np.sum(G_means * support) / total_samples
print("G_means macro avg: {:.4f}".format(g_means_macro_avg))
print("G_means weighted avg: {:.4f}".format(g_means_weighted_avg))
