import os
import argparse
import numpy as np
import pandas as pd

# 融合方法
def rank_mean(submissions):
    # 对每个提交文件的预测值进行排序
    ranked_submissions = []
    for sub in submissions:
        ranked = sub.rank()
        ranked_submissions.append(ranked)
    # 计算平均排名
    avg_rank = np.mean(ranked_submissions, axis=0)
    # 归一化到0-1
    min_rank = avg_rank.min()
    max_rank = avg_rank.max()
    return (avg_rank - min_rank) / (max_rank - min_rank)

def mean(submissions):
    # 直接平均
    return np.mean(submissions, axis=0)

def logit_mean(submissions):
    # 对数几率平均
    logits = np.log(np.array(submissions) / (1 - np.array(submissions)))
    avg_logits = np.mean(logits, axis=0)
    return 1 / (1 + np.exp(-avg_logits))

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='rank_mean', choices=['mean', 'rank_mean', 'logit_mean'], help='融合方法')
    parser.add_argument('--out', type=str, default='./submission_blend.csv', help='输出文件路径')
    parser.add_argument('--inputs', nargs='+', required=True, help='输入提交文件路径')
    args = parser.parse_args()
    
    print('加载提交文件...')
    submissions = []
    ids = None
    
    for input_file in args.inputs:
        df = pd.read_csv(input_file)
        if ids is None:
            ids = df['id'].values
        else:
            assert np.array_equal(ids, df['id'].values), '所有提交文件的ID必须一致'
        submissions.append(df['sentiment'].values)
    
    print('融合提交文件...')
    if args.method == 'mean':
        blended = mean(submissions)
    elif args.method == 'rank_mean':
        blended = rank_mean(submissions)
    elif args.method == 'logit_mean':
        blended = logit_mean(submissions)
    
    # 生成融合后的提交文件
    submission = pd.DataFrame({
        'id': ids,
        'sentiment': blended
    })
    submission.to_csv(args.out, index=False, quoting=3)
    print(f'融合后的提交文件已保存至: {args.out}')

if __name__ == '__main__':
    main()