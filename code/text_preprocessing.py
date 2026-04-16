import pandas as pd
import re
from bs4 import BeautifulSoup

# 内置停用词列表
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

# 简单的词干提取函数
def simple_stemmer(word):
    # 简单的词干提取规则
    suffixes = ['ing', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# 读取数据
def load_data(file_path):
    return pd.read_csv(file_path, sep='\t', escapechar='\\', quoting=3)

# 文本预处理函数
def preprocess_text(text):
    # 1. 移除 HTML 标记
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # 额外清理可能残留的 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 清理转义字符和反斜杠
    # 处理转义引号
    text = text.replace('\\"', '')
    text = text.replace('\\\'', '')
    
    # 清理所有反斜杠
    text = re.sub(r'\\+', '', text)
    
    # 3. 清理多余的引号
    text = re.sub(r'"+', '', text)
    
    # 4. 清理井号和其他特殊符号
    text = re.sub(r'[#*]+', '', text)
    
    # 5. 小写化
    text = text.lower()
    
    # 6. 再次清理反斜杠（确保无残留）
    text = re.sub(r'\\+', '', text)
    
    # 7. 处理标点符号和数字
    # 先将所有标点和数字替换为空格，然后移除多余的空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. 分词
    words = text.split()
    
    # 9. 移除停用词
    processed_words = []
    for word in words:
        # 清理单词中的反斜杠
        word = re.sub(r'\\+', '', word)
        if word not in stop_words:
            processed_words.append(word)
    
    # 10. 将单词重新连接成一个字符串，用空格分隔
    processed_text = ' '.join(processed_words)
    
    # 最终清理反斜杠
    processed_text = re.sub(r'\\+', '', processed_text)
    
    return processed_text

# 主函数
def main():
    # 读取数据
    file_path = 'labeledTrainData.tsv'
    df = load_data(file_path)
    
    # 预处理文本
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # 保存处理后的数据
    df.to_csv('processedTrainData.tsv', sep='\t', index=False)
    
    # 创建实验日志
    log_data = {
        'id': [20],
        '主要改动': ['移除词干提取步骤，保留完整单词，确保文本更加自然'],
        '实现结果': ['成功重新生成 processedTrainData.tsv 文件，确保文本完全干净无任何特殊符号且单词之间有正确的空格，保留完整单词形式']
    }
    log_df = pd.DataFrame(log_data)
    # 追加到现有日志文件
    try:
        existing_log = pd.read_csv('experiment_log.csv')
        log_df = pd.concat([existing_log, log_df], ignore_index=True)
    except FileNotFoundError:
        pass
    log_df.to_csv('experiment_log.csv', index=False)
    
    print("文本预处理完成，处理后的数据已保存到 processedTrainData.tsv")
    print("实验日志已保存到 experiment_log.csv")

if __name__ == "__main__":
    main()
