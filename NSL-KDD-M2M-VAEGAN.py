
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape, \
    GlobalAveragePooling1D, Flatten
# from keras.utils.np_utils import to_categorical
from sklearn import metrics
# from keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split

from util.ctgan.synthesizers.PreCVGMtvaeganConv1d import PRECVGMTVAEGANConv1d

import os
import matplotlib
matplotlib.use('TkAgg')  # 使用 Agg 后端，不显示图形窗口，可用于保存图片
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 后续代码保持不变
import torch
import numpy as np
import tensorflow as tf
import random as python_random
from keras.initializers import glorot_uniform

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

# Loading training set into dataframe
train_df = pd.read_csv('data/KDD/Training and Testing Sets/KDDTrain+.txt')
# Loading testing set into dataframe
test_df = pd.read_csv('data/KDD/Training and Testing Sets/KDDTest+.txt')


# Reset column names for training set
train_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                    'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']

#Reset column names for testing set
test_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']

# #去掉标签列
tmp = train_df.pop('subclass')
tmp1 = test_df.pop('subclass')

dos = ['mailbomb', 'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable']
probe = ['ipsweep', 'satan', 'nmap', 'portsweep', 'mscan', 'saint']
r2l = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop',
       'snmpguess', 'snmpgetattack', 'sendmail', 'named','worm']
u2r = ['buffer_overflow', 'loadmodule','httptunnel', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
classlist_test=[]
classlist_train = []

for i in range(0, len(tmp)):
    if tmp[i] == 'normal':
        classlist_train.append(0)
    elif tmp[i] in dos:
        classlist_train.append(1)
    elif tmp[i] in probe:
        classlist_train.append(2)
    elif tmp[i] in r2l:
        classlist_train.append(4)
    elif tmp[i] in u2r:
        classlist_train.append(3)

for i in range(0, len(tmp1)):
    if tmp1[i] == 'normal':
        classlist_test.append(0)
    elif tmp1[i] in dos:
        classlist_test.append(1)
    elif tmp1[i] in probe:
        classlist_test.append(2)
    elif tmp1[i] in r2l:
        classlist_test.append(4)
    elif tmp1[i] in u2r:
        classlist_test.append(3)

#Appending class column to training set
train_df["Class"] = classlist_train
test_df["Class"] = classlist_test
print(f"{train_df['Class'].value_counts()}")
print(f"测试数据分布: {test_df['Class'].value_counts()}")

orig_Data = pd.concat([train_df,test_df],ignore_index=True)


majority_classes = [1,2]
minority_classes = [3,4]
majority_train_data = train_df[train_df['Class'].isin(majority_classes)]
minority_train_data = train_df[train_df['Class'].isin(minority_classes)]
majority_train_data=majority_train_data.sample(n=10000)




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

# #Pretvaegan
loadpath1='data/KDD/PREVGMCVAEGANConv1d-majority-GM条件.pth'
loadpath2='data/KDD/PRECVAEGANConv1d-minority-GM条件250.pth'
discrete_columns = ['protocol_type', 'service', 'flag']

batch_size_u2r = 25
batch_size_r2l = 50
all_classes = [0,1, 2, 3, 4]#训练只有4个类型

random_seed = 42
synthesizer= PRECVGMTVAEGANConv1d(
    embedding_dim=128,
    l2scale=0.00001,
    loss_factor=0.1,
    cuda=True,
    verbose=False,
    c=0.005,
    discriminator_train_steps=5,
    num_class=len(all_classes)
)
# 预训练
synthesizer.set_random_state(random_seed)
majority_subset = majority_train_data.sample(frac=0.1, random_state=random_seed)
print(f"majority_subset分布：{majority_subset['Class'].value_counts()}")

tmp=majority_train_data.pop('Class')
majority_label=pd.get_dummies(tmp).astype(int)
# 确保独热编码后的列顺序与所有可能的类别顺序一致
for cls in all_classes:
    if cls not in majority_label.columns:
        majority_label[cls] = 0
majority_label = majority_label[all_classes]

synthesizer.fit(majority_train_data,majority_label,discrete_columns,batch_size=300,epoch=500, pretrain=True, finetune=False,savepath=loadpath1,pltpath='data/KDD/synthesizer-majority',dlr=0.00001,     lr2=0.0001, lr1=0.0001)
# 记录预训练性能
pretrain_time = time.time() - start_time
pretrain_mem = log_gpu_usage()['used_mb']
print(f"预训练耗时: {pretrain_time:.2f}s | 峰值内存: {pretrain_mem}MB")



#----------------------------------------------------------------------------------
print('微调')
# 在微调代码段修改如下：
start_time = time.time()
# 从多数类中随机抽取20%的数据
tmp=majority_subset.pop('Class')
majority_subset_label=pd.get_dummies(tmp).astype(int)
for cls in all_classes:
    if cls not in majority_subset_label.columns:
        majority_subset_label[cls] = 0
majority_subset_label = majority_subset_label[all_classes]

#r2l
tmp1 = minority_train_data.pop('Class')
minority_label_R2L = pd.get_dummies(tmp1).astype(int)
# 确保独热编码后的列顺序与所有可能的类别顺序一致
for cls in all_classes:
    if cls not in minority_label_R2L.columns:
        minority_label_R2L[cls] = 0
# 按照 all_classes 的顺序重新排列列
minority_label_R2L = minority_label_R2L[all_classes]

if os.path.exists(loadpath2):
    synthesizer = PRECVGMTVAEGANConv1d.load(loadpath2)
else:
    synthesizer.set_random_state(random_seed)
    # 合并少数类和多数类数据
    mixed_data = pd.concat([minority_train_data, majority_subset], ignore_index=True)
    #混合label
    mixed_labels = pd.concat([minority_label_R2L, majority_subset_label.loc[majority_subset.index]], ignore_index=True)
    synthesizer.fit(mixed_data, mixed_labels,discrete_columns,pretrain=False,batch_size=batch_size_r2l,epoch=250, finetune=True,pretrainpath=loadpath1, savepath=loadpath2,pltpath='data/KDD/pre-synthesizer_R2L',dlr=0.00001,lr1=0.000001,lr2=0.000001)
    synthesizer.save(loadpath2)

print('R2L处理')#
synth_data_R2L = synthesizer.sample(3000,batch_size=batch_size_r2l,majority_class_data=majority_train_data,num_class=len(all_classes),cond=(4-1))##由于训练时只有4个类型，因此采样时，需要在目标类别上，减少一个索引
synth_data_R2L['Class'] = 4


train_df_sys = pd.concat([train_df, synth_data_R2L], ignore_index=True)
#----------------------------------------------
print('U2R处理')
synth_data_U2R = synthesizer.sample(1000,batch_size=batch_size_u2r,majority_class_data=majority_train_data,num_class=len(all_classes),cond=(3-1))
synth_data_U2R['Class'] = 3
train_df_sys = pd.concat([train_df_sys, synth_data_U2R], ignore_index=True)
print(f"train_df_sys：{train_df_sys['Class'].value_counts()}")


# 记录微调性能
finetune_time = time.time() - start_time
finetune_mem = log_gpu_usage()['used_mb']
print(f"微调耗时: {finetune_time:.2f}s | 峰值内存: {finetune_mem}MB")
# 在最终结果输出部分添加：
print("\n=== 性能总结 ===")
print(f"预训练阶段: {pretrain_time:.2f}s | {pretrain_mem}MB")
print(f"微调阶段: {finetune_time:.2f}s | {finetune_mem}MB")
print(f"总耗时: {pretrain_time + finetune_time:.2f}s")


combined_data2 = pd.concat([train_df_sys,test_df],ignore_index=True)

cols = ['protocol_type','service','flag']
#One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, axis=1)
    return df
# 合并
combined_data2 = one_hot(combined_data2,cols)
tmp = combined_data2.pop('Class')
dfcolumns=combined_data2.columns
from sklearn.preprocessing import  MinMaxScaler
combined_data2 = MinMaxScaler().fit_transform(combined_data2)
combined_data2 = pd.DataFrame(combined_data2, columns=dfcolumns)
new_combined_data=pd.concat([combined_data2,tmp],axis=1)

train2_size = len(train_df_sys)
test_size = len(test_df)
# 拆分数据集
train_df_2 = new_combined_data.iloc[:train2_size]
test_df_2 = new_combined_data.iloc[train2_size:train2_size + test_size]


#划分数据集
y_train = train_df_2['Class']
y_test = test_df_2['Class']
X_train = train_df_2.drop('Class',  axis=1)
X_test = test_df_2.drop('Class',  axis=1)



#建立模型
cnn = Sequential()
cnn.add(Convolution1D(64, 3, padding="same",activation="relu",input_shape=(123, 1)))
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
cnn.add(Dense(5, activation="softmax", kernel_initializer=glorot_uniform(seed=42)))

# # #定义优化函数
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)  # 设置学习率为 0.0001
cnn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

#进一步将训练集拆分成训练和验证集   ?是否有必要
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)#101随机种子


#训练集: 据从二维数组（[样本数, 特征数]）重塑为三维数组（[样本数, 特征数, 1]）
x_columns_train = train_df_2.columns.drop('Class')
x_train_array = train_X[x_columns_train].values
x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
# x_train_1 = np.array(x_train_array)

dummies = pd.get_dummies(train_y)  #将分类标签（train_y 和 y_test）转换为独热编码（One-Hot Encoding）形式。
outcomes = dummies.columns
num_classes = len(outcomes)
y_train_1 = dummies.values#分类标签独热编码后的值

#验证集
x_columns_test = test_df_2.columns.drop('Class')
x_valid_array = test_X[x_columns_test].values
x_test_1 = np.reshape(x_valid_array, (x_valid_array.shape[0], x_valid_array.shape[1], 1))
# x_test_1 = np.array(x_valid_array)

dummies_test = pd.get_dummies(test_y)  # Classification
outcomes_test = dummies_test.columns
num_classes = len(outcomes_test)
y_test_1 = dummies_test.values

#测试集
x_columns_test = test_df_2.columns.drop('Class')
x_test_array = test_df_2[x_columns_test].values
x_test_2 = np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))
# x_test_2 = np.array(x_test_array)

dummies_test = pd.get_dummies(y_test)  # Classification
outcomes_test = dummies_test.columns
# num_classes = len(outcomes_test)
y_test_2 = dummies_test.values

history= cnn.fit(x_train_1, y_train_1,validation_data=(x_test_1,y_test_1), epochs=30,batch_size=64)

cnn.save('data/KDD/pre-条件-cnn.h5')  # HDF5格式
# ------------------ 下次使用时加载 -------------------
from tensorflow.keras.models import load_model
# 加载整个模型
# cnn = load_model('data/KDD/-cnn.h5')

# use matplitlib to draw the plots of last epoch

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
print("####   0:normal  1:Dos  2:Probe  3:U2R  4:R2L  ###\n\n")
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
g_means_macro_avg = np.mean(G_means)
g_means_weighted_avg = np.sum(G_means * support) / total_samples
print("G_means macro avg: {:.4f}".format(g_means_macro_avg))
print("G_means weighted avg: {:.4f}".format(g_means_weighted_avg))
