import pandas as pd
import re
from bs4 import BeautifulSoup

# 简单的词干提取函数
def simple_stemmer(word):
    suffixes = ['ing', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# 内置停用词列表（移除否定词）
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'ma'
])

# 保留的否定词
negative_words = {'not', 'no', 'nor', 'never', 'don', 'don\'t', 'ain', 'ain\'t', 'aren', 'aren\'t', 'couldn', 'couldn\'t', 'didn', 'didn\'t', 'doesn', 'doesn\'t', 'hadn', 'hadn\'t', 'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'mightn', 'mightn\'t', 'mustn', 'mustn\'t', 'needn', 'needn\'t', 'shan', 'shan\'t', 'shouldn', 'shouldn\'t', 'wasn', 'wasn\'t', 'weren', 'weren\'t', 'won', 'won\'t', 'wouldn', 'wouldn\'t'}

# 简单的词干提取函数（当NLTK不可用时使用）
def simple_stemmer(word):
    suffixes = ['ing', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# 读取数据
def load_data(file_path):
    try:
        return pd.read_csv(file_path, sep='\t', quoting=3)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

# 文本预处理函数
def preprocess_text(text):
    # 确保text是字符串
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 移除 HTML 标记
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # 额外清理可能残留的 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 清理转义字符和反斜杠
    text = text.replace('\\"', '')
    text = text.replace('\\\'', '')
    text = re.sub(r'\\+', '', text)
    
    # 3. 清理多余的引号
    text = re.sub(r'"+', '', text)
    
    # 4. 清理特殊符号
    text = re.sub(r'[#*]+', '', text)
    
    # 5. 清理其他特殊字符
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\\\\"\\|,.<>\/\?]', ' ', text)
    
    # 6. 小写化
    text = text.lower()
    
    # 7. 处理数字（移除）
    text = re.sub(r'\d+', '', text)
    
    # 8. 处理标点符号和多余的空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. 分词
    words = text.split()
    
    # 10. 移除停用词并进行词干提取
    processed_words = []
    for word in words:
        # 清理单词中的反斜杠
        word = re.sub(r'\\+', '', word)
        # 检查是否为停用词
        if word not in stop_words or word in negative_words:
            # 使用简单词干提取
            processed_word = simple_stemmer(word)
            processed_words.append(processed_word)
    
    # 11. 将单词重新连接成一个字符串，用空格分隔
    processed_text = ' '.join(processed_words)
    
    # 最终清理
    processed_text = re.sub(r'\\+', '', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

# 主函数
def main():
    # 读取数据
    file_path = 'labeledTrainData.tsv'
    print("正在读取数据...")
    df = load_data(file_path)
    
    if df is not None:
        print(f"数据读取成功，共 {len(df)} 条记录")
        
        # 预处理文本
        print("正在预处理文本...")
        df['processed_review'] = df['review'].apply(preprocess_text)
        
        # 保存处理后的数据
        output_path = 'processedTrainData.tsv'
        print(f"正在保存处理后的数据到 {output_path}...")
        df.to_csv(output_path, sep='\t', index=False, quoting=3)
        
        # 打印处理前后的样例
        print("\n处理前后的样例：")
        for i in range(min(3, len(df))):
            print(f"\n原始文本 (前500字符): {df['review'][i][:500]}...")
            print(f"处理后文本 (前500字符): {df['processed_review'][i][:500]}...")
        
        # 生成实验日志
        log_data = {
            'id': [38],
            '主要改动': ['重新开始文本预处理，按照要求清洗labeledTrainData.tsv文件：移除HTML标签、去除标点和特殊字符、转小写、处理数字、保留否定词、进行词形归并'],
            '实现结果': [f'成功生成processedTrainData.tsv文件，共处理 {len(df)} 条记录']
        }
        log_df = pd.DataFrame(log_data)
        
        # 追加到现有日志文件
        try:
            existing_log = pd.read_csv('results/experiment_log.csv')
            log_df = pd.concat([existing_log, log_df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        log_df.to_csv('results/experiment_log.csv', index=False, encoding='utf-8')
        print("\n实验日志已更新到 results/experiment_log.csv")
        
        print("\n文本预处理完成！")
    else:
        print("数据读取失败，无法进行预处理")

if __name__ == "__main__":
    main()
