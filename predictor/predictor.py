import json
import thulac
#from sklearn.externals import joblib
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder



class Predictor(object):
	def __init__(self):
		# 移除 TF-IDF 模型加载，因为它将被词向量替换
		# self.tfidf = joblib.load('predictor/model/tfidf.model')

		# 加载自己训练的 Word2Vec 模型
		self.word2vec_model = Word2Vec.load('predictor/model/my_word2vec.model')

		# 其他模型保持加载，但请记住：这些模型需要重新训练！
		self.law = joblib.load('predictor/model/law.model')
		self.accu = joblib.load('predictor/model/accu.model')
		self.time = joblib.load('predictor/model/time.model')

		# 加载 MultiLabelBinarizer 和 LabelEncoder 实例
		self.mlb_accu = joblib.load('predictor/model/accu_mlb.model')
		self.mlb_law = joblib.load('predictor/model/law_mlb.model')
		self.le_time = joblib.load('predictor/model/time_le.model')

		self.batch_size = 1
		
		self.cut = thulac.thulac(seg_only = True)

	def _get_sentence_vector(self, words):
		"""
		将分词后的词语列表转换为句向量（通过词向量平均）
		处理未知词（OOV）和空句子
		"""
		# 直接从模型的 wv 属性中获取词向量
		vector_size = self.word2vec_model.wv.vector_size 
		
		valid_word_vectors = []
		for word in words:
			# 检查词语是否在模型的词汇表中
			if word in self.word2vec_model.wv:
				valid_word_vectors.append(self.word2vec_model.wv[word])
		
		if len(valid_word_vectors) == 0:
			# 如果句子中没有词在词向量模型中，返回一个零向量
			return np.zeros(vector_size, dtype=np.float32)
		else:
			# 对所有已知词的词向量求平均
			return np.mean(valid_word_vectors, axis=0)

	def _flatten_segmented_words(self, raw_segmented_words):
		"""
		辅助函数：将 thulac 分词结果扁平化为纯字符串列表。
		处理 thulac 可能返回列表的列表或元组列表的情况。
		"""
		flattened_words = []
		for item in raw_segmented_words:
			if isinstance(item, list) and len(item) > 0:
				flattened_words.append(str(item[0]))
			elif isinstance(item, str):
				flattened_words.append(item)
			elif isinstance(item, tuple) and len(item) > 0:
				flattened_words.append(str(item[0]))
		return flattened_words

	def predict_law(self, vec):
		# 预测结果是二进制矩阵，需要逆转换
		y_pred_bin = self.law.predict(vec)
		# inverse_transform 期望一个二维数组，即使只有一个样本
		y_transformed = self.mlb_law.inverse_transform(y_pred_bin)
		# inverse_transform 返回的是一个列表的列表，例如 [[1, 2]]，我们需要提取里面的列表
		return list(y_transformed[0]) # 确保返回的是纯 Python list，不是 numpy.ndarray
	
	def predict_accu(self, vec):
		# 预测结果是二进制矩阵，需要逆转换
		y_pred_bin = self.accu.predict(vec)
		y_transformed = self.mlb_accu.inverse_transform(y_pred_bin)
		return list(y_transformed[0]) # 确保返回的是纯 Python list
	
	def predict_time(self, vec):
		# 刑期模型现在是分类器，返回编码后的整数
		y_encoded = self.time.predict(vec)[0] # predict 返回数组，取第一个元素
		
		# 将编码后的整数逆转换回原始的刑期值
		# le_time.inverse_transform 期望一个一维数组，即使只有一个元素
		original_time_value = self.le_time.inverse_transform([y_encoded])[0]

		# 保持原来的特殊值映射逻辑，因为它更明确
		# 注意：这里假设 LabelEncoder 会将 -2, -1 也作为类别进行编码
		# 如果 LabelEncoder 训练时没有这些值，需要特别处理
		# 我们的 train_classifiers.py 中已经将 -2, -1 也纳入了 LabelEncoder
		return int(original_time_value)

	def predict(self, content):
		# 首先，使用 thulac 进行分词，获取原始分词结果
		raw_fact_cut_lists = [self.cut.cut(x, text=False) for x in content]

		# 然后，对每个分词结果进行扁平化处理
		fact_cut_lists = [self._flatten_segmented_words(raw_words) for raw_words in raw_fact_cut_lists]

		# 将每个分词后的文本（词语列表）转换为句向量
		vecs = [self._get_sentence_vector(word_list) for word_list in fact_cut_lists]

		# 将列表转换为 numpy 数组，以便 joblib 模型可以处理
		# 形状将是 (样本数量, 词向量维度)
		vec_array = np.array(vecs)

		results = []
		for i in range(len(content)):
			ans = {}
			# joblib 模型通常期望输入是 (n_samples, n_features) 的二维数组
			# 因此对于单个样本，需要 reshape 成 (1, -1)
			single_vec = vec_array[i].reshape(1, -1) 
			
			ans['accusation'] = self.predict_accu(single_vec)
			ans['articles'] = self.predict_law(single_vec)
			ans['imprisonment'] = self.predict_time(single_vec)
			results.append(ans)

		return results

		 
