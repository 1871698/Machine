import pandas as pd
import re
import os

# 读取unlabeledTrain.tsv文件
def load_unlabeled_data(file_path):
    """读取未标记的训练数据"""
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return None
    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    return df

# 使用正则表达式进行句子拆分
def split_sentences(text):
    """使用正则表达式拆分句子"""
    try:
        # 简单的句子拆分规则：以句号、问号、感叹号结尾的句子
        sentences = re.split(r'[.!?]+', text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    except Exception as e:
        print(f"句子拆分失败: {e}")
        return []

# 主函数
def main():
    # 文件路径
    unlabeled_data_path = 'unlabeledTrainData.tsv'
    
    # 读取未标记的训练数据
    print("正在读取unlabeledTrainData.tsv文件...")
    df = load_unlabeled_data(unlabeled_data_path)
    
    if df is not None:
        # 统计数据量
        print(f"\n数据量: {len(df)} 条评论")
        
        # 对每条评论进行句子拆分
        print("\n正在使用punkt分词器进行句子拆分...")
        all_sentences = []
        # 处理所有评论
        total_reviews = len(df)
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"处理到第 {i} 条评论")
            text = row.get('review', '')
            sentences = split_sentences(text)
            all_sentences.extend(sentences)
        
        # 统计句子数量
        print(f"\n句子拆分完成，总句子数: {len(all_sentences)}")
        
        # 查看前10个句子
        print("\n前10个句子:")
        for i, sentence in enumerate(all_sentences[:10]):
            print(f"{i+1}. {sentence}")
        
        # 写入实验日志
        print("\n正在写入实验日志...")
        log_data = {
            'id': [23],
            '主要改动': ['引入分布式词向量，使用正则表达式对unlabeledTrainData.tsv文件的所有评论进行句子拆分'],
            '实现结果': [f'成功读取unlabeledTrainData.tsv文件，处理所有{total_reviews}条评论，拆分出{len(all_sentences)}个句子']
        }
        log_df = pd.DataFrame(log_data)
        # 追加到现有日志文件
        try:
            existing_log = pd.read_csv('experiment_log.csv')
            log_df = pd.concat([existing_log, log_df], ignore_index=True)
        except FileNotFoundError:
            pass
        log_df.to_csv('experiment_log.csv', index=False, encoding='utf-8')
        print("实验日志已更新")
    else:
        print("无法读取数据文件，实验失败")

if __name__ == "__main__":
    main()