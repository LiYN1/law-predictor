import json
import thulac
from gensim.models import Word2Vec
import os
import multiprocessing

# 定义数据路径和模型保存路径
DATA_TRAIN_PATH = '../data_train.json'
MODEL_SAVE_PATH = '../predictor/model/my_word2vec.model'  # 将新模型保存在 predictor/model 目录下

# 初始化 thulac 分词器（注意：thulac 实例不能直接在多进程间共享，每个进程需要自己的实例）
# cut = thulac.thulac(seg_only=True) # 移除全局定义

def init_worker():
    """
    为每个进程初始化 thulac 实例，因为 thulac 对象不能被pickle
    """
    global worker_cut
    worker_cut = thulac.thulac(seg_only=True)

def process_fact(fact_text):
    """
    每个 worker 进程处理一个 fact 文本的分词，并确保结果是扁平的字符串列表
    """
    if fact_text:
        raw_segmented_words = worker_cut.cut(fact_text, text=False)
        
        # 扁平化处理：确保返回的是一个字符串列表，而不是列表的列表
        # 例如：[['word1'], ['word2']] 变为 ['word1', 'word2']
        # 如果 thulac 返回的是 [('word1', 'pos1'), ('word2', 'pos2')]，也提取出词语
        flattened_words = []
        for item in raw_segmented_words:
            if isinstance(item, list) and len(item) > 0:
                flattened_words.append(str(item[0]))
            elif isinstance(item, str): # 如果 thulac 已经返回字符串列表，直接添加
                flattened_words.append(item)
            elif isinstance(item, tuple) and len(item) > 0: # 如果是 (word, pos) 元组，取第一个元素
                flattened_words.append(str(item[0]))
        return flattened_words
    return []

def get_sentences_from_json_parallel(json_file_path):
    """
    从 JSON 文件中读取 'fact' 字段，并行进行分词，并返回句子列表（每个句子是词语列表）
    """
    fact_texts = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                fact_text = data.get('fact', '')
                if fact_text:
                    fact_texts.append(fact_text)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line.strip()} - {e}")
                continue
    
    print(f"开始并行分词 {len(fact_texts)} 条文本...")
    # 使用 multiprocessing.Pool 进行并行处理
    # initializer 和 initargs 用于为每个进程初始化 thulac 实例
    # os.cpu_count() 获取可用的 CPU 核心数
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
        sentences = list(pool.map(process_fact, fact_texts))
    
    return [s for s in sentences if s] # 过滤掉可能为空的句子

def train_and_save_word2vec_model():
    print(f"开始从 {DATA_TRAIN_PATH} 读取数据并分词 (并行处理)...")
    sentences = get_sentences_from_json_parallel(DATA_TRAIN_PATH) # 调用并行处理函数
    print(f"分词完成，共获取 {len(sentences)} 条句子。")

    if not sentences:
        print("没有获取到任何句子，请检查 data_train.json 文件内容和路径。")
        return

    print("开始训练 Word2Vec 模型...")
    # 参数解释：
    # vector_size: 词向量的维度，通常取 100, 200, 300
    # window: 训练时考虑的上下文词语数量
    # min_count: 忽略总频率低于此值的词语
    # workers: 训练时使用的线程数（CPU 核心数）
    # sg: 0 代表 CBOW 模型，1 代表 Skip-gram 模型 (Skip-gram 在大数据集和不常见词上表现更好)
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=os.cpu_count(), sg=1)
    
    # 保存模型，使用 gensim 自己的保存格式
    model.save(MODEL_SAVE_PATH)
    print(f"Word2Vec 模型训练完成并保存到 {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # 在 Windows 上，多进程代码需要包裹在 if __name__ == '__main__': 中
    # 我们已经确保了所有多进程相关逻辑都在这个块内
    train_and_save_word2vec_model() 