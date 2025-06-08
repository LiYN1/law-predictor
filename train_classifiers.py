import json
import thulac
import joblib
import os
import numpy as np
from gensim.models import Word2Vec # 用于加载训练好的词向量模型


from sklearn.svm import LinearSVC, SVR # 用于罪名/法条和刑期预测
from sklearn.multiclass import OneVsRestClassifier # 多标签分类可能需要
from sklearn.preprocessing import MultiLabelBinarizer # 标签二值化器
from sklearn.preprocessing import LabelEncoder # 用于刑期标签编码

import multiprocessing

# 定义数据路径和模型保存路径
DATA_TRAIN_PATH = 'data_train.json'
WORD2VEC_MODEL_PATH = 'predictor/model/my_word2vec.model'

LAW_MODEL_SAVE_PATH = 'predictor/model/law.model'
ACCU_MODEL_SAVE_PATH = 'predictor/model/accu.model'
TIME_MODEL_SAVE_PATH = 'predictor/model/time.model'

# MultiLabelBinarizer 实例的保存路径
ACCU_MLB_SAVE_PATH = ACCU_MODEL_SAVE_PATH.replace('.model', '_mlb.model')
LAW_MLB_SAVE_PATH = LAW_MODEL_SAVE_PATH.replace('.model', '_mlb.model')
# 刑期标签编码器的保存路径
TIME_LE_SAVE_PATH = TIME_MODEL_SAVE_PATH.replace('.model', '_le.model')

ACCUSATION_PATH = 'accu.txt'  # 用于获取罪名列表，转换标签
LAW_PATH = 'law.txt'  # 用于获取法条列表，转换标签

# 全局变量，用于在多进程工作器中加载模型
worker_cut = None
worker_word2vec_model = None

def init_worker():
    """
    多进程池工作器的初始化函数。
    每个工作器进程将初始化自己的 thulac 实例并加载 Word2Vec 模型一次。
    """
    global worker_cut
    global worker_word2vec_model
    worker_cut = thulac.thulac(seg_only=True)
    worker_word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

def _flatten_segmented_words(raw_segmented_words):
    flattened_words = []
    for item in raw_segmented_words:
        if isinstance(item, list) and len(item) > 0:
            flattened_words.append(str(item[0]))
        elif isinstance(item, str):
            flattened_words.append(item)
        elif isinstance(item, tuple) and len(item) > 0:
            flattened_words.append(str(item[0]))
    return flattened_words

def _get_sentence_vector(word2vec_model, words):
    vector_size = word2vec_model.wv.vector_size
    valid_word_vectors = []
    for word in words:
        if word in word2vec_model.wv:
            valid_word_vectors.append(word2vec_model.wv[word])
    
    if len(valid_word_vectors) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    else:
        return np.mean(valid_word_vectors, axis=0)

def process_single_fact_and_vectorize(fact_text):
    """
    工作器函数，用于分词和向量化单个事实文本。
    使用全局的 worker_cut 和 worker_word2vec_model。
    """
    if not fact_text:
        # 如果 fact_text 为空，返回零向量（与 Word2Vec 模型维度匹配）
        if worker_word2vec_model is not None: # 确保模型已加载
            return np.zeros(worker_word2vec_model.wv.vector_size, dtype=np.float32)
        return np.array([]) # 否则返回空数组或根据实际情况处理

    raw_segmented_words = worker_cut.cut(fact_text, text=False)
    flat_words = _flatten_segmented_words(raw_segmented_words)
    sentence_vec = _get_sentence_vector(worker_word2vec_model, flat_words)
    return sentence_vec

def load_data_and_extract_features_parallel(json_file_path):
    all_fact_texts = []
    all_meta_data = []

    # 在主进程中顺序读取所有数据和元数据
    print(f"[主进程] 正在从 {json_file_path} 读取所有数据...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                fact_text = data.get('fact', '')
                meta = data.get('meta', {})
                if fact_text and meta: # 只处理有 fact 和 meta 的条目
                    all_fact_texts.append(fact_text)
                    all_meta_data.append(meta)
            except json.JSONDecodeError as e:
                print(f"[主进程] 解码 JSON 行时出错: {line.strip()} - {e}")
                continue
    print(f"[主进程] 已读取 {len(all_fact_texts)} 条数据。")

    print(f"[主进程] 开始并行处理 {len(all_fact_texts)} 条文本的特征提取...")
    # 使用 multiprocessing Pool 进行特征提取，每个工作器加载自己的 thulac 和 Word2Vec 实例
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
        features = list(pool.map(process_single_fact_and_vectorize, all_fact_texts))
    print(f"[主进程] 并行特征提取完成。")
    # 在主进程中顺序处理标签
    law_labels = []
    accu_labels = []
    time_labels = []

    # 加载罪名和法条的映射（在主进程中加载一次即可）
    accu_map = {}
    with open(ACCUSATION_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            accu_map[line.strip()] = i + 1
    
    law_map = {}
    with open(LAW_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            law_map[int(line.strip())] = i + 1

    print("[主进程] 开始处理标签...")
    for meta in all_meta_data:
        # 提取罪名标签 (可能多标签)
        current_accu_labels = []
        for acc_name in meta.get('accusation', []):
            acc_name_clean = acc_name.replace("[", "").replace("]", "")
            if acc_name_clean in accu_map:
                current_accu_labels.append(accu_map[acc_name_clean])
        accu_labels.append(current_accu_labels)

        # 提取法条标签 (可能多标签)
        current_law_labels = []
        for law_id in meta.get('relevant_articles', []):
            if law_id in law_map:
                current_law_labels.append(law_map[law_id])
        law_labels.append(current_law_labels)

        # 提取刑期标签 (回归问题)
        imprisonment_info = meta.get('term_of_imprisonment', {})
        if imprisonment_info.get('death_penalty'):
            time_labels.append(-2) # -2 表示死刑
        elif imprisonment_info.get('life_imprisonment'):
            time_labels.append(-1) # -1 表示无期徒刑
        else:
            # 将刑期统一到月份
            time_labels.append(imprisonment_info.get('imprisonment', 0))

    print("[主进程] 标签处理完成。")

    # 将列表转换为 numpy 数组
    X = np.array(features)
    y_law = law_labels
    y_accu = accu_labels
    y_time = np.array(time_labels)

    return X, y_law, y_accu, y_time, accu_map, law_map


def train_models():
    # Word2Vec 模型在每个工作器中加载，所以这里不再需要全局加载
    print("[主进程] 开始从 {} 提取特征和标签 (并行处理)...".format(DATA_TRAIN_PATH))
    X, y_law, y_accu, y_time, accu_map, law_map = load_data_and_extract_features_parallel(DATA_TRAIN_PATH)
    print(f"[主进程] 特征和标签提取完成。样本数: {len(X)}, 特征维度: {X.shape[1]}")

    # --- 训练罪名预测模型 (Accusation Model) ---
    print("[主进程] 开始训练罪名预测模型...")
    mlb_accu = MultiLabelBinarizer(classes=range(1, len(accu_map) + 1))
    y_accu_bin = mlb_accu.fit_transform(y_accu)
    
    accu_classifier = OneVsRestClassifier(LinearSVC(random_state=42, max_iter=10000)) 
    accu_classifier.fit(X, y_accu_bin)
    joblib.dump(accu_classifier, ACCU_MODEL_SAVE_PATH)
    joblib.dump(mlb_accu, ACCU_MLB_SAVE_PATH) # 使用新的MLB保存路径
    print(f"[主进程] 罪名预测模型和MultiLabelBinarizer训练完成并保存到 {ACCU_MODEL_SAVE_PATH} 和 {ACCU_MLB_SAVE_PATH}")
    
    # --- 训练法条预测模型 (Law Model) ---
    print("[主进程] 开始训练法条预测模型...")
    mlb_law = MultiLabelBinarizer(classes=range(1, len(law_map) + 1))
    y_law_bin = mlb_law.fit_transform(y_law)

    law_classifier = OneVsRestClassifier(LinearSVC(random_state=42, max_iter=10000))
    law_classifier.fit(X, y_law_bin)
    joblib.dump(law_classifier, LAW_MODEL_SAVE_PATH)
    joblib.dump(mlb_law, LAW_MLB_SAVE_PATH) # 使用新的MLB保存路径
    print(f"[主进程] 法条预测模型和MultiLabelBinarizer训练完成并保存到 {LAW_MODEL_SAVE_PATH} 和 {LAW_MLB_SAVE_PATH}")

    # --- 训练刑期预测模型 (Time Model) ---
    print("[主进程] 开始训练刑期预测模型...")
    # 首先，定义刑期的所有可能的离散值，包括特殊值
    # 这些值需要与 predictor.py 中的映射保持一致
    # 假设这些是模型需要区分的类别
    unique_time_labels = sorted(list(set(y_time))) # 从数据中找出所有唯一的刑期值

    # 过滤掉不参与训练的特殊值（例如，如果特殊值不作为分类目标）
    # 这里我们认为-2和-1也是分类目标
    # 所以，将所有time_labels进行编码
    le_time = LabelEncoder()
    # fit_transform 会将 unique_time_labels 映射到 0, 1, 2...的整数
    y_time_encoded = le_time.fit_transform(y_time)
    
    # 刑期现在是多分类问题
    time_classifier = LinearSVC(random_state=42, max_iter=10000) # 或者 SVC
    time_classifier.fit(X, y_time_encoded)
    joblib.dump(time_classifier, TIME_MODEL_SAVE_PATH)
    joblib.dump(le_time, TIME_LE_SAVE_PATH) # 保存 LabelEncoder
    print(f"[主进程] 刑期预测模型和LabelEncoder训练完成并保存到 {TIME_MODEL_SAVE_PATH} 和 {TIME_LE_SAVE_PATH}")

    print("[主进程] 所有模型训练和保存完成！")

if __name__ == '__main__':
    train_models() 