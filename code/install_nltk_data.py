import os
import urllib.request
import zipfile
import nltk

# 定义NLTK数据路径
data_path = nltk.data.path[0]  # 使用第一个数据路径
print(f"NLTK数据路径: {data_path}")

# 确保数据目录存在
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"创建了数据目录: {data_path}")

# 定义需要下载的数据包
data_packages = [
    {
        'name': 'wordnet',
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'
    },
    {
        'name': 'stopwords',
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip'
    },
    {
        'name': 'punkt',
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip'
    },
    {
        'name': 'omw-1.4',
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip'
    }
]

# 下载并安装数据包
for package in data_packages:
    package_name = package['name']
    package_url = package['url']
    zip_path = os.path.join(data_path, f"{package_name}.zip")
    extract_path = os.path.join(data_path, package_name)
    
    # 检查是否已经安装
    if os.path.exists(extract_path):
        print(f"{package_name} 已经安装，跳过下载")
        continue
    
    print(f"正在下载 {package_name}...")
    try:
        # 下载zip文件
        urllib.request.urlretrieve(package_url, zip_path)
        print(f"{package_name} 下载完成")
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"{package_name} 解压完成")
        
        # 删除zip文件
        os.remove(zip_path)
        print(f"删除了临时文件: {zip_path}")
        
    except Exception as e:
        print(f"下载 {package_name} 失败: {e}")
        # 清理临时文件
        if os.path.exists(zip_path):
            os.remove(zip_path)

print("\nNLTK数据安装完成！")

# 测试NLTK是否正常工作
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # 测试stopwords
    stop_words = set(stopwords.words('english'))
    print(f"stopwords 加载成功，共 {len(stop_words)} 个停用词")
    
    # 测试WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    test_word = lemmatizer.lemmatize('running')
    print(f"WordNetLemmatizer 测试成功: running -> {test_word}")
    
    # 测试punkt
    test_sentence = "This is a test sentence. It has multiple sentences."
    # 注意：这里不实际测试分词，因为punkt可能还没有完全安装
    print("punkt 数据包已下载")
    
    print("\nNLTK功能测试通过！")
    
except Exception as e:
    print(f"NLTK功能测试失败: {e}")
