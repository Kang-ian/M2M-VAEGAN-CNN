"""CLI."""
import pandas as pd

import argparse

from util.ctgan.data import read_csv, read_tsv, write_tsv
# from util.ctgan.synthesizers.ctgan import CTGAN
from sdv.single_table import CTGANSynthesizer

def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    #周期
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    #如果设置，数据将从TSV（Tab-Separated Values）格式加载，而不是CSV。
    parser.add_argument(
        '-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV'
    )
    # 如果CSV文件没有标题行，将离散列作为索引处理。
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )
    #元数据文件的路径。
    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    # 描述：没有空格的离散列的逗号分隔列表。
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    #要采样的行数。默认为训练数据的大小。
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Number of rows to sample. Defaults to the training data size',
    )
    #生成器的学习率。
    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )
    #权重衰减
    parser.add_argument(
        '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.'
    )
    parser.add_argument(
        '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.'
    )

    parser.add_argument(
        '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.'
    )
    parser.add_argument(
        '--generator_dim',
        type=str,
        default='256,256',
        help='Dimension of each generator layer. ' 'Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--discriminator_dim',
        type=str,
        default='256,256',
        help='Dimension of each discriminator layer. '
        'Comma separated integers with no whitespaces.',
    )
    #批量大小。必须是偶数。
    parser.add_argument(
        '--batch_size', type=int, default=500, help='Batch size. Must be an even number.'
    )
    # 3保存训练好的合成器的文件名。
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    # 3加载训练好的合成器的文件名。
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )
    #选择一个离散列的名称。
    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    #指定所选离散列的值。
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument('data/preprocessed/perfectdata-train.csv.gz', help='Path to training data')
    parser.add_argument('data/preprocessed/ctgan.csv.gz', help='Path of the output file')

    return parser.parse_args()


def main():
    """CLI."""
    # args = _parse_args()
    # if args.tsv:
    #     data, discrete_columns = read_tsv(args.data, args.metadata)
    # else:D:\PycharmDatas\Transformer\RTIDS\data\preprocessed
    #     data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)
    import os

    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_path = os.path.join(current_dir, '../../data/preprocessed/')
    data = pd.read_csv(os.path.join(preprocessed_path, 'perfectdata-train.csv.gz'), compression='gzip')
    print(data['Label'].value_counts())
    # 使用随机抽样获取四分之一的数据
    data = data.sample(frac=0.005, random_state=42)  # 设置random_state可确保每次抽样结果一致（便于重现结果）
    print(data['Label'].value_counts())
    test_data = pd.read_csv(os.path.join(preprocessed_path, 'perfectdata-train.csv.gz'), compression='gzip')

    # test_data = pd.read_csv('D:/PycharmDatas/Transformer/RTIDS/data/preprocessed/test_data.csv.gz', compression='gzip')
    print(test_data['Label'].value_counts())
    test_data = test_data.sample(frac=0.005, random_state=42)  # 设置random_state可确保每次抽样结果一致（便于重现结果）
    print(data['Label'].value_counts())
    discrete_columns= ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count', 'SYN Flag Count',
                'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
                'ECE Flag Count', 'Label', 'Attack', 'BENIGN']

    synthesizer = CTGANSynthesizer(
        discrete_columns,  # required
        enforce_rounding=False,
        epochs=500,
        verbose=True
    )

    synthesizer.fit(data)

    synthesizer.save(
        filepath='my_synthesizer.pkl'
    )

    synthetic_data = synthesizer.sample(num_rows=10)





















    # if args.load:
    #     model = CTGAN.load(args.load)
    # else:
    # generator_dim = [int(x) for x in args.generator_dim.split(',')]
    # discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
    model = CTGAN(
        embedding_dim=128,
        generator_dim=[256,256],
        discriminator_dim=[256,256],
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        epochs=30,
        )
    # model.fit(data, discrete_columns)
    # model.save('data/preprocessed/ctgan.pth')
    if not os.path.exists("../../data/preprocessed/ctgan.pth"):
        print('开始fit')
        model.fit(data, discrete_columns)
        model.save('../../data/preprocessed/ctgan.pth')
    else:
        model.load('../../data/preprocessed/ctgan.pth')
    num_samples = 1500

    # if args.sample_condition_column is not None:
    #     assert args.sample_condition_column_value is not None

    sampled = model.sample(3000,condition_column='Label', condition_value=3)

    # if args.tsv:
    #     write_tsv(sampled, args.metadata, args.output)
    # else:
    #     sampled.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
